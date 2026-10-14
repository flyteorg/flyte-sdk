/*
 * AutoSec demo target — SECURE (no known vulnerability).
 *
 * `greet()` bounds the copy with strncpy() to sizeof(name)-1 and always
 * NUL-terminates, so an over-long argv[1] is safely truncated rather than
 * overflowing the buffer.
 */
#include <stdio.h>
#include <string.h>

void greet(const char *input) {
    char name[64];
    strncpy(name, input, sizeof(name) - 1);
    name[sizeof(name) - 1] = '\0'; /* bounded copy + explicit termination */
    printf("Hello, %s!\n", name);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <name>\n", argv[0]);
        return 1;
    }
    greet(argv[1]);
    return 0;
}
