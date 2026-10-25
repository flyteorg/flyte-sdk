/*
 * AutoSec demo target — DELIBERATELY VULNERABLE.
 *
 * Planted bug: `copy_record()` trusts strlen(input) as the memcpy length into a
 * fixed 40-byte buffer. There is no check that the length fits, so a long
 * argv[1] overflows `record` (a length/bounds confusion).
 */
#include <stdio.h>
#include <string.h>

void copy_record(const char *input) {
    char record[40];
    size_t n = strlen(input);
    memcpy(record, input, n); /* VULN: n unchecked against sizeof(record) */
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
