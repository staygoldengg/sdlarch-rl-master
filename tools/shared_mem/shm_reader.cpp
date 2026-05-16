/**
 * shm_reader.cpp — Shared memory reader (the "consumer" process).
 *
 * Opens the named mapping created by shm_writer and reads the integer.
 * The writer process must already be running (it owns the mapping lifetime).
 *
 * Build (MSVC):  cl /O2 shm_reader.cpp /link kernel32.lib
 * Build (MinGW): g++ -O2 shm_reader.cpp -o shm_reader.exe -lkernel32
 */
#include <windows.h>
#include <iostream>

static constexpr wchar_t MAP_NAME[] = L"Local\\MySharedInt";
static constexpr DWORD   MAP_SIZE   = sizeof(int);

int main() {
    // OpenFileMapping fails if the writer hasn't created it yet.
    HANDLE hMap = OpenFileMappingW(FILE_MAP_READ, FALSE, MAP_NAME);

    if (!hMap) {
        std::cerr << "[reader] OpenFileMapping failed: " << GetLastError()
                  << "  (is the writer process running?)\n";
        return 1;
    }

    const int* pVal = static_cast<const int*>(
        MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, MAP_SIZE)
    );

    if (!pVal) {
        std::cerr << "[reader] MapViewOfFile failed: " << GetLastError() << "\n";
        CloseHandle(hMap);
        return 1;
    }

    std::cout << "[reader] Read value: " << *pVal << "\n";

    UnmapViewOfFile(pVal);
    CloseHandle(hMap);
    return 0;
}
