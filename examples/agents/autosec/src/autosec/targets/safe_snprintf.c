/*
 * AutoSec demo target — SECURE (no known vulnerability).
 *
 * `format_label()` uses snprintf() with the destination size, so the write is
 * bounded by sizeof(label) regardless of how long argv[1] is.
 */
#include <stdio.h>

void format_label(const char *input) {
    char label[32];
    snprintf(label, sizeof(label), "label=%s", input); /* size-bounded write */
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
