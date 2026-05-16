/**
 * shm_writer.cpp — Shared memory writer (the "producer" process).
 *
 * Creates a named shared memory region and writes a single integer into it.
 * The reader process can open the same name and read the value.
 *
 * Build (MSVC):  cl /O2 shm_writer.cpp /link kernel32.lib
 * Build (MinGW): g++ -O2 shm_writer.cpp -o shm_writer.exe -lkernel32
 *
 * Usage: shm_writer.exe [integer_value]
 *   e.g. shm_writer.exe 99
 */
#include <windows.h>
#include <iostream>
#include <cstdlib>

// Both processes must use the same name.
// "Local\" scope = same user session. Use "Global\" for cross-session (needs SeCreateGlobalPrivilege).
static constexpr wchar_t MAP_NAME[] = L"Local\\MySharedInt";
static constexpr DWORD   MAP_SIZE   = sizeof(int);

int main(int argc, char* argv[]) {
    int value = (argc > 1) ? std::atoi(argv[1]) : 42;

    HANDLE hMap = CreateFileMappingW(
        INVALID_HANDLE_VALUE,   // backed by the page file (no file on disk)
        nullptr,                // default security
        PAGE_READWRITE,
        0, MAP_SIZE,
        MAP_NAME
    );

    if (!hMap) {
        std::cerr << "[writer] CreateFileMapping failed: " << GetLastError() << "\n";
        return 1;
    }

    int* pVal = static_cast<int*>(
        MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, MAP_SIZE)
    );

    if (!pVal) {
        std::cerr << "[writer] MapViewOfFile failed: " << GetLastError() << "\n";
        CloseHandle(hMap);
        return 1;
    }

    *pVal = value;
    std::cout << "[writer] Wrote " << value << " to \"" << "Local\\MySharedInt" << "\".\n";
    std::cout << "[writer] Press Enter to release the mapping and exit...\n";
    std::cin.get();

    UnmapViewOfFile(pVal);
    CloseHandle(hMap);
    return 0;
}
