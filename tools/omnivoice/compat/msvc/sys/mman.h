/* Minimal read-only mmap shim for MSVC / Windows.
 *
 * The vendored voice-classifier sources (wakeword / voice_classifier / vad)
 * map their GGUF files read-only — always
 *   mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0)  /  munmap(addr, size)
 * — so this shim implements exactly that surface over the Win32 file-mapping
 * API. It is only on the include path for MSVC builds (see the eliza_voice_
 * classifiers target in tools/omnivoice/CMakeLists.txt); POSIX hosts use the
 * real <sys/mman.h>.
 */
#ifndef ELIZA_COMPAT_MSVC_SYS_MMAN_H
#define ELIZA_COMPAT_MSVC_SYS_MMAN_H

#include <io.h> /* _get_osfhandle */
#include <stddef.h>
#include <windows.h>

#define PROT_NONE 0x0
#define PROT_READ 0x1
#define PROT_WRITE 0x2

#define MAP_SHARED 0x1
#define MAP_PRIVATE 0x2

#define MAP_FAILED ((void *)-1)

static __inline void *mmap(void *addr, size_t length, int prot, int flags,
                           int fd, long offset) {
  (void)addr;
  (void)prot;
  (void)flags;
  HANDLE file = (HANDLE)_get_osfhandle(fd);
  if (file == INVALID_HANDLE_VALUE) {
    return MAP_FAILED;
  }
  HANDLE mapping = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL);
  if (mapping == NULL) {
    return MAP_FAILED;
  }
  void *view = MapViewOfFile(mapping, FILE_MAP_READ, 0, (DWORD)offset, length);
  /* The mapped view holds a reference to the mapping object, so the handle can
   * be released immediately; munmap() then only needs the base address. */
  CloseHandle(mapping);
  return view ? view : MAP_FAILED;
}

static __inline int munmap(void *addr, size_t length) {
  (void)length;
  return UnmapViewOfFile(addr) ? 0 : -1;
}

#endif /* ELIZA_COMPAT_MSVC_SYS_MMAN_H */
