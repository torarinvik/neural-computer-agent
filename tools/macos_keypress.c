/* Minimal macOS key-event transport for the replaceable output backend.
 *
 * This program has no application, task, or semantic knowledge. It accepts
 * one external virtual key code, emits a key-down/key-up pair, and exits.
 * The Python boundary performs target identity and frontmost-window checks
 * before invoking it.
 */

#include <ApplicationServices/ApplicationServices.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static int emit_key(const char *text) {
    char *end = NULL;
    long value;
    CGEventRef down;
    CGEventRef up;

    errno = 0;
    value = strtol(text, &end, 10);
    while (*end == '\n' || *end == '\r') end++;
    if (errno != 0 || end == text || *end != '\0' || value < 0 || value > USHRT_MAX) {
        fprintf(stderr, "invalid macOS virtual key code\n");
        return 2;
    }
    down = CGEventCreateKeyboardEvent(NULL, (CGKeyCode)value, true);
    up = CGEventCreateKeyboardEvent(NULL, (CGKeyCode)value, false);
    if (down == NULL || up == NULL) {
        if (down != NULL) CFRelease(down);
        if (up != NULL) CFRelease(up);
        fprintf(stderr, "could not allocate keyboard events\n");
        return 1;
    }
    CGEventPost(kCGHIDEventTap, down);
    CGEventPost(kCGHIDEventTap, up);
    CFRelease(down);
    CFRelease(up);
    return 0;
}

int main(int argc, char **argv) {
    char line[64];
    int result;

    if (argc == 2) return emit_key(argv[1]);
    if (argc != 1) {
        fprintf(stderr, "usage: macos_keypress [KEY_CODE]\n");
        return 2;
    }
    while (fgets(line, sizeof(line), stdin) != NULL) {
        result = emit_key(line);
        if (result != 0) return result;
        if (fputc('.', stdout) == EOF || fflush(stdout) != 0) return 1;
    }
    return 0;
}
