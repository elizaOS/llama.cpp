/* Minimal <unistd.h> shim for MSVC / Windows.
 *
 * The vendored voice-classifier sources use open()/close()/read() and ssize_t
 * from <unistd.h>, which MSVC does not ship. MSVC exposes the POSIX I/O names
 * (open/close/read/lseek) from <io.h> under _CRT_DECLARE_NONSTDC_NAMES (the
 * default), and ssize_t maps to the Win32 SSIZE_T. Only on the MSVC include
 * path (see tools/omnivoice/CMakeLists.txt); POSIX hosts use the real header.
 */
#ifndef ELIZA_COMPAT_MSVC_UNISTD_H
#define ELIZA_COMPAT_MSVC_UNISTD_H

#include <basetsd.h> /* SSIZE_T */
#include <io.h>      /* open, close, read, lseek (POSIX names) */
#include <process.h>

#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#endif

#endif /* ELIZA_COMPAT_MSVC_UNISTD_H */
