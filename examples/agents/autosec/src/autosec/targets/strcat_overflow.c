/*
 * AutoSec demo target — DELIBERATELY VULNERABLE.
 *
 * Planted bug: `build_path()` seeds a 48-byte stack buffer with a prefix and
 * then strcat()s an attacker-controlled suffix onto it with no remaining-space
 * check. A long argv[1] overflows `path`.
 */
#include <stdio.h>
#include <string.h>

void build_path(const char *suffix) {
    char path[48];
    strcpy(path, "/var/data/");
    strcat(path, suffix); /* VULN: no check that suffix fits in remaining space */
    printf("opening %s\n", path);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <suffix>\n", argv[0]);
        return 1;
    }
    build_path(argv[1]);
    return 0;
}
