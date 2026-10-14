/*
 * AutoSec demo target — DELIBERATELY VULNERABLE.
 *
 * Planted bug: `format_label()` writes an attacker-controlled string into a
 * 32-byte stack buffer with sprintf() and a "%s" format. Any argv[1] longer
 * than 31 bytes overflows `label`.
 */
#include <stdio.h>

void format_label(const char *input) {
    char label[32];
    sprintf(label, "label=%s", input); /* VULN: unbounded sprintf into small buffer */
    printf("%s\n", label);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <value>\n", argv[0]);
        return 1;
    }
    format_label(argv[1]);
    return 0;
}
