/*
 * AutoSec demo target — SECURE (no known vulnerability).
 *
 * `copy_record()` uses memcpy() — which static analysis flags as dangerous —
 * but clamps the length to sizeof(record)-1 first. This is a deliberate
 * false-positive case: a flagged call site that is actually safe.
 */
#include <stdio.h>
#include <string.h>

void copy_record(const char *input) {
    char record[40];
    size_t n = strlen(input);
    if (n >= sizeof(record)) {
        n = sizeof(record) - 1; /* clamp before copy -> no overflow */
    }
    memcpy(record, input, n);
    record[n] = '\0';
    printf("record: %s\n", record);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <data>\n", argv[0]);
        return 1;
    }
    copy_record(argv[1]);
    return 0;
}
