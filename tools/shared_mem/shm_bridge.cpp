// tools/shared_mem/shm_bridge.cpp
//
// SharedMemoryBridge — Windows named shared memory segment that carries
// live Brawlhalla state from a writer process (game reader / memory scanner)
// to the Python weaponized_ai stack via ctypes/cffi.
//
// Build (MSVC):
//   cl /O2 /LD shm_bridge.cpp /Fe:shm_bridge.dll
//
// Build (MinGW):
//   g++ -O2 -shared -o shm_bridge.dll shm_bridge.cpp
//
// Python usage (ctypes):
//   lib = ctypes.CDLL("shm_bridge.dll")
//   bridge = lib.CreateBridge()
//   lib.WriteBridgeState(bridge, ctypes.byref(state_buf))
//   lib.DestroyBridge(bridge)

#include <windows.h>
#include <cstring>
#include <cstdint>
#include <iostream>

#define SHM_POOL_SIZE 4096
#define SHM_NAME      "Local\\StrikerEnlightenedBridge"

// ── State buffer layout ───────────────────────────────────────────────────────
// #pragma pack(push, 1) ensures no compiler padding so byte offsets are stable
// across the Python struct.unpack / np.frombuffer reader in watchdog_reader.py.
//
// Byte offsets (matches TelemetryWatchdogMemoryReader):
//   0   – 7:   alignment_checksum  (uint64)
//   8   – 11:  current_frame       (uint32)
//   12  – 15:  player_x            (float)
//   16  – 19:  player_y            (float)
//   20  – 23:  opponent_x          (float)
//   24  – 27:  opponent_y          (float)
//   28  – 31:  player_damage       (float)
//   32  – 35:  opponent_damage     (float)
//   36  – 39:  player_stocks       (uint32)
//   40  – 43:  opponent_stocks     (uint32)
//   44  – 235: feature_vector[48]  (48 × float)
#pragma pack(push, 1)
struct BrawlhallaStateBuffer {
    uint64_t alignment_checksum;
    uint32_t current_frame;
    float    player_x;
    float    player_y;
    float    opponent_x;
    float    opponent_y;
    float    player_damage;
    float    opponent_damage;
    uint32_t player_stocks;
    uint32_t opponent_stocks;
    float    feature_vector[48];
};
#pragma pack(pop)

static_assert(sizeof(BrawlhallaStateBuffer) == 236,
              "BrawlhallaStateBuffer size mismatch — check struct layout");

// ── Bridge class ──────────────────────────────────────────────────────────────
class SharedMemoryBridge {
private:
    HANDLE                hMapFile = nullptr;
    BrawlhallaStateBuffer* pBuffer  = nullptr;

public:
    bool Initialize() {
        hMapFile = CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            nullptr,
            PAGE_READWRITE,
            0,
            SHM_POOL_SIZE,
            SHM_NAME
        );

        if (hMapFile == nullptr) {
            std::cerr << "[CRITICAL] SHM Mapping failed: " << GetLastError() << "\n";
            return false;
        }

        pBuffer = reinterpret_cast<BrawlhallaStateBuffer*>(
            MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, SHM_POOL_SIZE)
        );

        if (pBuffer == nullptr) {
            CloseHandle(hMapFile);
            hMapFile = nullptr;
            return false;
        }

        // Sentinel checksum so readers can verify the segment is initialised
        pBuffer->alignment_checksum = 0xDEADC0DEFA57FEEDULL;
        return true;
    }

    void PushState(const BrawlhallaStateBuffer& freshState) {
        if (pBuffer != nullptr) {
            // Full struct copy — no partial-write tearing risk for atomic-sized fields
            std::memcpy(pBuffer, &freshState, sizeof(BrawlhallaStateBuffer));
        }
    }

    void Close() {
        if (pBuffer)  { UnmapViewOfFile(pBuffer);  pBuffer  = nullptr; }
        if (hMapFile) { CloseHandle(hMapFile);     hMapFile = nullptr; }
    }
};

// ── C Bindings ────────────────────────────────────────────────────────────────
// Flat C ABI so Python ctypes can call directly without a C++ name mangling dance.
extern "C" {

    __declspec(dllexport) SharedMemoryBridge* CreateBridge() {
        auto* bridge = new SharedMemoryBridge();
        if (bridge->Initialize()) return bridge;
        delete bridge;
        return nullptr;
    }

    __declspec(dllexport) void WriteBridgeState(
        SharedMemoryBridge*    bridge,
        BrawlhallaStateBuffer* state
    ) {
        if (bridge && state) bridge->PushState(*state);
    }

    __declspec(dllexport) void DestroyBridge(SharedMemoryBridge* bridge) {
        if (bridge) {
            bridge->Close();
            delete bridge;
        }
    }

} // extern "C"
