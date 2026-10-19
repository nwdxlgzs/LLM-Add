#include "cairn.h"
#include <string.h>
#include <stdint.h>
static const char *const specials[5] = {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"};
size_t cn_encode(const void *bytes, size_t length, int parse, int *out, size_t cap) {
    const unsigned char *p = (const unsigned char *)bytes;
    size_t pos = 0, n = 0;
    if ((!p && length) || (!out && cap))
        return SIZE_MAX;
    while (pos < length) {
        int id = p[pos];
        size_t width = 1;
        if (parse && p[pos] == '<')
            for (int j = 0; j < 5; j++) {
                size_t k = strlen(specials[j]);
                if (k <= length - pos && !memcmp(p + pos, specials[j], k)) {
                    id = 256 + j;
                    width = k;
                    break;
                }
            }
        if (n < cap)
            out[n] = id;
        ++n;
        pos += width;
    }
    return n;
}
size_t cn_decode(const int *ids, size_t count, int render, void *out, size_t cap) {
    unsigned char *p = (unsigned char *)out;
    size_t n = 0;
    if ((!ids && count) || (!out && cap))
        return SIZE_MAX;
    for (size_t i = 0; i < count; i++) {
        int id = ids[i];
        if (id < 0 || id >= CN_VOCAB)
            return SIZE_MAX;
        if (id < 256) {
            if (n < cap)
                p[n] = (unsigned char)id;
            ++n;
        } else if (render) {
            size_t k = strlen(specials[id - 256]);
            if (k > SIZE_MAX - n)
                return SIZE_MAX;
            for (size_t j = 0; j < k; j++)
                if (n + j < cap)
                    p[n + j] = (unsigned char)specials[id - 256][j];
            n += k;
        }
    }
    return n;
}
