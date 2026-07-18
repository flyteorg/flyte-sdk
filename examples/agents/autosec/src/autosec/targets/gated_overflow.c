/*
 * AutoSec demo target — VULNERABLE, but resistant to the naive PoC.
 *
 * `handle()` has a genuine unbounded strcpy into a 64-byte stack buffer, so a
 * vulnerability researcher (and the model) will flag it. BUT the vulnerable
 * copy is only reached when the input begins with the "DEBUG:" prefix. The
 * demo's generic proof-of-concept just sends a long run of 'A's as argv[1],
 * which never enters the vulnerable branch — so the bug is real yet the PoC
 * fails to trigger a crash. This lands in the amber "VULNERABLE" (flagged, not
 * triggered) state, illustrating the gap between a hypothesis and a PoC that
 * doesn't account for reachability conditions.
 */
#include <stdio.h>
#include <string.h>

void handle(const char *input) {
    char buf[64];
    if (strncmp(input, "DEBUG:", 6) == 0) {
        strcpy(buf, input); /* VULN: unbounded, but gated behind the DEBUG: prefix */
        printf("debug: %s\n", buf);
        return;
    }
    printf("ok\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <command>\n", argv[0]);
        return 1;
    }
    handle(argv[1]);
    return 0;
}
