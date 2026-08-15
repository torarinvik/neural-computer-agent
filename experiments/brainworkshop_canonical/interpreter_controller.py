"""The controller as interpreter: programs are data it reads, not Python.

`DECISION_CONTROLLER_IS_THE_INTERPRETER.md` makes this normative. The counter
bridge reached 18/18 with a Python executor deciding every press while the
controller never ran; that is a ceiling, and this is the path.

The design follows the invariants in that decision, and each one shows up as a
concrete mechanism rather than a promise:

- **instructions are opaque data.** An instruction is one row of program
  tensor. Nothing in the controller enumerates instruction types.
- **operators are content-addressed.** The program carries a handle table. The
  controller emits an intention; the runtime selects whichever handle that
  intention is nearest to. Adding an operator adds a row to the table and
  leaves the controller's parameters and digest untouched — which is the whole
  reason the decoder does not get one output per opcode.
- **workspace is external.** Slots live beside the program, not in the
  network, so more working memory never resizes the controller.
- **budgets fail closed.** A tick that exhausts its microstep budget records
  that status and emits nothing. It never falls back to a default action.

Execution is structural; the learned part is only the choice of intention. To
keep those separable this module can run in two modes. `teacher` reads the
intention from the instruction itself, which exercises the machinery without
any trained network and is how behaviour preservation is checked. `learned`
asks the controller, and is meaningless until the controller has been
pretrained to interpret — which it has not been yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

INTERPRETER_PROGRAM_SCHEMA = "neural-computer.interpreted-program.v1"
INTERPRETER_STATUS = ("halted", "budget_exhausted", "invalid_operand")

# Operator identities are opaque to the controller: they are rows in a table
# the program carries. These names exist for records and tests, never as
# controller features.
OPERATOR_NAMES = ("advance", "jump", "store", "emit_match", "emit", "halt")


@dataclass(frozen=True)
class InterpretedProgram:
    """Instruction rows, an operator handle table, and workspace geometry."""

    handles: torch.Tensor          # [operators, event_width]
    operators: tuple[str, ...]     # names, for records only
    instructions: torch.Tensor     # [rows, event_width] opaque instruction data
    operator_index: tuple[int, ...]  # which handle each row names
    operands: tuple[int, ...]      # one integer operand per row
    workspace_slots: int
    microstep_budget: int = 64
    schema: str = INTERPRETER_PROGRAM_SCHEMA

    def validate(self) -> InterpretedProgram:
        if self.schema != INTERPRETER_PROGRAM_SCHEMA:
            raise ValueError("unsupported interpreted program schema")
        if self.handles.ndim != 2 or self.handles.shape[0] < 1:
            raise ValueError("an interpreted program needs a handle table")
        if self.instructions.ndim != 2:
            raise ValueError("instructions must be a row tensor")
        rows = self.instructions.shape[0]
        if rows < 1:
            raise ValueError("an interpreted program needs at least one row")
        width = int(self.handles.shape[1])
        if self.instructions.shape[1] not in (width, 2 * width):
            raise ValueError(
                "instruction width must be one or two handle widths: a single "
                "field names an operator outright, a pair names one for a met "
                "condition and one for an unmet condition"
            )
        if len(self.operator_index) != rows or len(self.operands) != rows:
            raise ValueError("every row needs an operator and an operand")
        if len(self.operators) != self.handles.shape[0]:
            raise ValueError("operator names must cover the handle table")
        if any(not 0 <= index < self.handles.shape[0] for index in self.operator_index):
            raise ValueError("instruction names an operator outside the table")
        if self.workspace_slots < 1 or self.microstep_budget < 1:
            raise ValueError("workspace and budget must be positive")
        return self

    @property
    def event_width(self) -> int:
        return int(self.handles.shape[1])

    def with_operator(self, name: str, handle: torch.Tensor) -> InterpretedProgram:
        """Add an operator. The controller must be unaffected by this."""

        return InterpretedProgram(
            handles=torch.cat((self.handles, handle.reshape(1, -1)), dim=0),
            operators=(*self.operators, name),
            instructions=self.instructions,
            operator_index=self.operator_index,
            operands=self.operands,
            workspace_slots=self.workspace_slots,
            microstep_budget=self.microstep_budget,
        ).validate()


def operator_handles(event_width: int, *, seed: int, count: int = len(OPERATOR_NAMES)) -> torch.Tensor:
    """Opaque, reproducible handles. Their values carry no meaning."""

    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        table = torch.randn(count, event_width)
    return table / table.norm(dim=1, keepdim=True)


class InterpreterController(nn.Module):
    """Maps (event, instruction, workspace) to an intention. Nothing more.

    The controller never names an operator. It emits an intention into the
    same space the handles live in, and the runtime resolves it by proximity,
    so the operator vocabulary is a property of the program rather than of
    these parameters.
    """

    def __init__(self, event_width: int, *, hidden: int = 64) -> None:
        super().__init__()
        self.event_width = int(event_width)
        # event, primary field, alternate field, workspace summary, and the
        # elementwise difference that a condition turns on.
        self.read = nn.Sequential(
            nn.Linear(event_width * 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, event_width),
        )

    def forward(
        self,
        event: torch.Tensor,
        instruction: torch.Tensor,
        workspace_summary: torch.Tensor,
    ) -> torch.Tensor:
        width = self.event_width
        if instruction.shape[-1] == width:
            # A single field names its operator whatever the condition says.
            instruction = torch.cat((instruction, instruction), dim=-1)
        joined = torch.cat(
            (
                event,
                instruction,
                workspace_summary,
                (event - workspace_summary).abs(),
            ),
            dim=-1,
        )
        intention = self.read(joined)
        return intention / intention.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def digest(self) -> str:
        import hashlib

        hasher = hashlib.sha256()
        for name, parameter in sorted(self.named_parameters()):
            hasher.update(name.encode())
            hasher.update(parameter.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()


def resolve_operator(intention: torch.Tensor, handles: torch.Tensor) -> int:
    """Content addressing: the nearest handle wins."""

    return int((handles @ intention.reshape(-1)).argmax().item())


@dataclass
class TickResult:
    press: int | None
    status: str
    microsteps: int


def run_tick(
    program: InterpretedProgram,
    controller: InterpreterController | None,
    event: torch.Tensor,
    workspace: torch.Tensor,
    *,
    mode: str = "teacher",
    match_tolerance: float = 0.5,
) -> TickResult:
    """Interpret one environment tick, fail-closed on budget exhaustion."""

    program.validate()
    if mode not in ("teacher", "learned"):
        raise ValueError("interpreter mode must be teacher or learned")
    if mode == "learned" and controller is None:
        raise ValueError("learned mode needs a controller")
    pointer = 0
    press: int | None = None
    for step in range(program.microstep_budget):
        if not 0 <= pointer < program.instructions.shape[0]:
            return TickResult(None, "invalid_operand", step)
        instruction = program.instructions[pointer]
        if mode == "teacher":
            # The instruction names its own operator; this exercises the
            # machinery without asking an untrained network to interpret.
            operator = int(program.operator_index[pointer])
        else:
            summary = workspace.mean(dim=0)
            intention = controller(
                event.reshape(1, -1),
                instruction.reshape(1, -1),
                summary.reshape(1, -1),
            )
            operator = resolve_operator(intention, program.handles)
        name = program.operators[operator]
        operand = int(program.operands[pointer])
        if name == "halt":
            return TickResult(press, "halted", step + 1)
        if name == "advance":
            pointer += 1
        elif name == "jump":
            if not 0 <= operand < program.instructions.shape[0]:
                return TickResult(None, "invalid_operand", step + 1)
            pointer = operand
        elif name == "store":
            if not 0 <= operand < workspace.shape[0]:
                return TickResult(None, "invalid_operand", step + 1)
            workspace[operand] = event.reshape(-1)
            pointer += 1
        elif name == "emit_match":
            if not 0 <= operand < workspace.shape[0]:
                return TickResult(None, "invalid_operand", step + 1)
            distance = torch.linalg.vector_norm(workspace[operand] - event.reshape(-1))
            press = int(float(distance) <= match_tolerance)
            pointer += 1
        elif name == "emit":
            press = int(operand != 0)
            pointer += 1
        else:
            return TickResult(None, "invalid_operand", step + 1)
    # Budget exhausted: record it and emit nothing rather than defaulting.
    return TickResult(None, "budget_exhausted", program.microstep_budget)


def one_back_program(event_width: int, *, seed: int, budget: int = 64) -> InterpretedProgram:
    """A program, as data, for the capability the leases already verified.

    Press when the current event matches the one stored last tick, then store
    the current event for the next tick. Written here as an experimenter's
    reference program to check that interpretation preserves behaviour; it is
    not learned and is not admitted.
    """

    handles = operator_handles(event_width, seed=seed)
    names = OPERATOR_NAMES
    rows = ("emit_match", "store", "halt")
    operands = (0, 0, 0)
    # Instructions carry their operator's handle in both fields: this program
    # is unconditional, so the met and unmet cases name the same operator.
    chosen = torch.stack([handles[names.index(row)] for row in rows])
    instructions = torch.cat((chosen, chosen), dim=1)
    return InterpretedProgram(
        handles=handles,
        operators=names,
        instructions=instructions,
        operator_index=tuple(names.index(row) for row in rows),
        operands=operands,
        workspace_slots=1,
        microstep_budget=budget,
    ).validate()
