/*
 * AutoSec demo target — DELIBERATELY VULNERABLE.
 *
 * Planted bug: `greet()` copies an attacker-controlled argument into a fixed
 * 64-byte stack buffer with strcpy(), with no bounds check. Any argv[1] longer
 * than 63 bytes overflows `name` and smashes the stack.
 */
#include <stdio.h>
#include <string.h>

void greet(const char *input) {
    char name[64];
    strcpy(name, input); /* VULN: no bounds check, classic stack overflow */
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
