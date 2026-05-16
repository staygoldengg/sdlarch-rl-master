// shm_client.ts
// TypeScript side of the shared memory bridge for a Tauri app.
//
// Architecture:
//   TypeScript (this file) ──invoke──► Tauri Rust command
//                                         └──► OpenFileMapping / MapViewOfFile
//
// Add the corresponding Rust command to src-tauri/src/lib.rs (see below).
// Then call readSharedInt() / writeSharedInt() from any component.

import { invoke } from '@tauri-apps/api/core'

/** Read the integer from the shared memory region. */
export async function readSharedInt(mapName = 'Local\\\\MySharedInt'): Promise<number> {
  return await invoke<number>('shm_read_int', { mapName })
}

/** Write an integer into the shared memory region (writer side). */
export async function writeSharedInt(value: number, mapName = 'Local\\\\MySharedInt'): Promise<void> {
  await invoke<void>('shm_write_int', { mapName, value })
}

// ─────────────────────────────────────────────────────────────────────────────
// Rust commands to add to src-tauri/src/lib.rs:
// ─────────────────────────────────────────────────────────────────────────────
//
// use windows::Win32::System::Memory::*;
// use windows::core::PCWSTR;
// use std::ffi::OsStr;
// use std::os::windows::ffi::OsStrExt;
//
// fn wide(s: &str) -> Vec<u16> {
//     OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
// }
//
// #[tauri::command]
// fn shm_read_int(map_name: &str) -> Result<i32, String> {
//     let name = wide(map_name);
//     unsafe {
//         let h = OpenFileMappingW(FILE_MAP_READ.0, false, PCWSTR(name.as_ptr()))
//             .map_err(|e| e.to_string())?;
//         let ptr = MapViewOfFile(h, FILE_MAP_READ, 0, 0, 4);
//         if ptr.Value.is_null() { return Err("MapViewOfFile failed".into()); }
//         let val = *(ptr.Value as *const i32);
//         UnmapViewOfFile(ptr);
//         CloseHandle(h);
//         Ok(val)
//     }
// }
//
// #[tauri::command]
// fn shm_write_int(map_name: &str, value: i32) -> Result<(), String> {
//     let name = wide(map_name);
//     unsafe {
//         let h = CreateFileMappingW(
//             INVALID_HANDLE_VALUE, None, PAGE_READWRITE, 0, 4,
//             PCWSTR(name.as_ptr())
//         ).map_err(|e| e.to_string())?;
//         let ptr = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, 4);
//         if ptr.Value.is_null() { return Err("MapViewOfFile failed".into()); }
//         *(ptr.Value as *mut i32) = value;
//         UnmapViewOfFile(ptr);
//         CloseHandle(h);
//         Ok(())
//     }
// }
//
// Register in builder: .invoke_handler(tauri::generate_handler![shm_read_int, shm_write_int])
