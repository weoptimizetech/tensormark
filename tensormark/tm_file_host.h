// File-backed weight views. Offset-based reads and mapping are host services;
// the TMQ parser and tensor lifetime rules are identical on both platforms.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <vector>
#elif defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#error "tm_file_host.h requires a supported file mapping backend"
#endif

namespace tmfile {

#if defined(_WIN32)
inline int open_read(const std::string& path) {
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                          path.c_str(), -1, nullptr, 0);
    if (count <= 0) return -1;
    std::wstring wide(static_cast<std::size_t>(count), L'\0');
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                             path.c_str(), -1, wide.data(), count)) return -1;
    // A seek-independent ReadFile needs an overlapped handle. The CRT fd owns
    // and closes the HANDLE, including when a mapped view outlives the fd.
    HANDLE h = CreateFileW(wide.c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(h), _O_RDONLY | _O_BINARY);
    if (fd < 0) CloseHandle(h);
    return fd;
}
inline void close(int fd) { (void)_close(fd); }
inline bool size(int fd, std::size_t& bytes) {
    LARGE_INTEGER length{};
    const auto h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE || !GetFileSizeEx(h, &length) ||
        length.QuadPart < 8 ||
        static_cast<std::uint64_t>(length.QuadPart) >
            static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) return false;
    bytes = static_cast<std::size_t>(length.QuadPart);
    return true;
}
inline std::ptrdiff_t pread(int fd, void* dst, std::size_t n, std::uint64_t off) {
    if (n > std::numeric_limits<DWORD>::max()) n = std::numeric_limits<DWORD>::max();
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(off);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ov.hEvent) return -1;
    DWORD read = 0;
    const auto h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    const bool ok = h != INVALID_HANDLE_VALUE &&
        (ReadFile(h, dst, static_cast<DWORD>(n), &read, &ov) ||
         (GetLastError() == ERROR_IO_PENDING && GetOverlappedResult(h, &ov, &read, TRUE)));
    CloseHandle(ov.hEvent);
    return ok ? static_cast<std::ptrdiff_t>(read) : -1;
}
inline void* map(int fd, std::size_t n, std::uint64_t off) {
    const auto h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE || !n) return nullptr;
    HANDLE section = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!section) return nullptr;
    void* view = MapViewOfFile(section, FILE_MAP_READ,
                               static_cast<DWORD>(off >> 32), static_cast<DWORD>(off), n);
    CloseHandle(section);  // The view owns a reference until unmap.
    return view;
}
inline void unmap(void* p, std::size_t) { (void)UnmapViewOfFile(p); }
inline bool lock(const void* p, std::size_t n) {
    if (!p || !n) return false;
    // Windows defaults to a tiny lockable working set, regardless of free RAM.
    // VirtualLock otherwise refuses real model mappings with error 1453. Grow
    // the process minimum BEFORE locking, leaving room for the engine's own
    // pages; the OS may still refuse the quota, which remains a load failure.
    constexpr std::size_t headroom = std::size_t{256} << 20;
    if (n > std::numeric_limits<std::size_t>::max() - headroom) {
        SetLastError(ERROR_ARITHMETIC_OVERFLOW);
        return false;
    }
    const HANDLE process = GetCurrentProcess();
    SIZE_T minimum = 0, maximum = 0;
    if (!GetProcessWorkingSetSize(process, &minimum, &maximum)) return false;
    const SIZE_T required = n + headroom;
    if (minimum < required &&
        !SetProcessWorkingSetSize(process, required, std::max(maximum, required)))
        return false;
    return VirtualLock(const_cast<void*>(p), n) != 0;
}
inline void unlock(const void* p, std::size_t n) { (void)VirtualUnlock(const_cast<void*>(p), n); }
inline std::size_t alignment() {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwAllocationGranularity;  // MapViewOfFile offset, not page size
}
inline std::uint64_t physical_ram() {
    MEMORYSTATUSEX mem{sizeof(mem)};
    return GlobalMemoryStatusEx(&mem) ? mem.ullTotalPhys : 0;
}
inline void prefetch(const void* p, std::size_t n) {
    if (!p || !n) return;
    WIN32_MEMORY_RANGE_ENTRY range{const_cast<void*>(p), n};
    (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
}
inline void prefetch_file(int fd, std::uint64_t off, std::size_t n) {
    // No F_RDADVISE for an unmapped range: synchronously warm the file cache
    // rather than silently dropping the caller's prefetch request.
    std::vector<std::uint8_t> scratch(1u << 20);
    while (n) {
        const auto r = pread(fd, scratch.data(), std::min(n, scratch.size()), off);
        if (r <= 0) break;
        off += static_cast<std::uint64_t>(r);
        n -= static_cast<std::size_t>(r);
    }
}
inline std::string lock_error() { return "Windows error " + std::to_string(GetLastError()); }
inline constexpr const char* lock_name() { return "VirtualLock"; }

#else
inline int open_read(const std::string& path) { return ::open(path.c_str(), O_RDONLY); }
inline void close(int fd) { (void)::close(fd); }
inline bool size(int fd, std::size_t& bytes) {
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size < 8 ||
        static_cast<std::uintmax_t>(st.st_size) >
            static_cast<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max())) return false;
    bytes = static_cast<std::size_t>(st.st_size);
    return true;
}
inline std::ptrdiff_t pread(int fd, void* dst, std::size_t n, std::uint64_t off) {
    return ::pread(fd, dst, n, static_cast<off_t>(off));
}
inline void* map(int fd, std::size_t n, std::uint64_t off) {
    void* p = ::mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(off));
    return p == MAP_FAILED ? nullptr : p;
}
inline void unmap(void* p, std::size_t n) { (void)::munmap(p, n); }
inline bool lock(const void* p, std::size_t n) { return ::mlock(p, n) == 0; }
inline void unlock(const void* p, std::size_t n) { (void)::munlock(p, n); }
inline std::size_t alignment() { return static_cast<std::size_t>(::sysconf(_SC_PAGESIZE)); }
inline std::uint64_t physical_ram() {
    const long pages = ::sysconf(_SC_PHYS_PAGES);
    const long page_size = ::sysconf(_SC_PAGESIZE);
    return pages > 0 && page_size > 0
        ? static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size) : 0;
}
inline void prefetch(const void* p, std::size_t n) { (void)::madvise(const_cast<void*>(p), n, MADV_WILLNEED); }
inline void prefetch_file(int fd, std::uint64_t off, std::size_t n) {
    while (n) {
        const auto chunk = std::min(n, static_cast<std::size_t>(std::numeric_limits<int>::max()));
        struct radvisory rv{static_cast<off_t>(off), static_cast<int>(chunk)};
        if (::fcntl(fd, F_RDADVISE, &rv) != 0) break;
        off += chunk;
        n -= chunk;
    }
}
inline std::string lock_error() { return std::strerror(errno); }
inline constexpr const char* lock_name() { return "mlock"; }
#endif
}  // namespace tmfile
