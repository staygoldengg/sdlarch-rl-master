/**
 * replay.cpp — Offline input log player for local physics testing.
 *
 * Reads input_log.txt (or a path given as argv[1]) and replays each entry
 * through Windows SendInput at the recorded timestamps.
 *
 * Log file format (one entry per line, comments start with #):
 *   <timestamp_ms>  <scancode_hex>  <DOWN|UP>
 *   Example:
 *     0     0x1E   DOWN    // A key down at t=0
 *     50    0x1E   UP      // A key up  at t=50ms
 *     80    0x20   DOWN    // D key down at t=80ms
 *
 * Build (MSVC):  cl /O2 replay.cpp /link winmm.lib user32.lib
 * Build (MinGW): g++ -O2 replay.cpp -o replay.exe -lwinmm -luser32
 */
#include <windows.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "user32.lib")

struct Entry {
    DWORD timestamp_ms;
    WORD  scancode;
    bool  key_up;
};

static void fire(WORD scan, bool up) {
    INPUT inp      = {};
    inp.type       = INPUT_KEYBOARD;
    inp.ki.wScan   = scan;
    inp.ki.dwFlags = KEYEVENTF_SCANCODE | (up ? KEYEVENTF_KEYUP : 0);
    SendInput(1, &inp, sizeof(INPUT));
}

int main(int argc, char* argv[]) {
    const char* path = (argc > 1) ? argv[1] : "input_log.txt";

    std::ifstream f(path);
    if (!f) {
        std::cerr << "[replay] Cannot open: " << path << "\n";
        return 1;
    }

    std::vector<Entry> log;
    std::string line;
    int lineNum = 0;

    while (std::getline(f, line)) {
        ++lineNum;
        // Strip leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;

        std::istringstream ss(line.substr(start));
        DWORD ts;
        WORD  sc;
        std::string dir;

        if (!(ss >> std::dec >> ts >> std::hex >> sc >> dir)) {
            std::cerr << "[replay] Skipping malformed line " << lineNum << ": " << line << "\n";
            continue;
        }

        bool up = (dir == "UP" || dir == "up");
        log.push_back({ ts, sc, up });
    }

    if (log.empty()) {
        std::cerr << "[replay] No valid entries found in " << path << "\n";
        return 1;
    }

    std::cout << "[replay] Loaded " << log.size() << " entries. Starting playback...\n";

    // Request 1 ms system timer resolution for accurate Sleep(1) behaviour
    timeBeginPeriod(1);

    DWORD t0 = timeGetTime();

    for (const auto& e : log) {
        DWORD target = t0 + e.timestamp_ms;
        DWORD now;

        // Coarse sleep until we're within 2 ms of the target,
        // then spin the remaining distance to avoid overshooting.
        while ((now = timeGetTime()) < target) {
            DWORD remaining = target - now;
            if (remaining > 2)
                Sleep(1);
            // else: busy-spin the last ≤2 ms for precision
        }

        fire(e.scancode, e.key_up);
    }

    timeEndPeriod(1);
    std::cout << "[replay] Playback complete.\n";
    return 0;
}
