// The maintained engine lives at the checkout root. Source distributions stage
// an identical copy under _native/ without adding a second maintained source.
#pragma once
#if __has_include("_native/neural_demo.cpp")
#include "_native/neural_demo.cpp"
#else
#include "../neural_demo.cpp"
#endif
