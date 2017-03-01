#pragma once
#include <functional>
#include <chrono>

// Time the execution of f for the given number of iterations
inline std::chrono::duration<double> time_func(std::function<bool()> f, int iterations)
{
	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < iterations; ++i)
	{
		if (!f())
			break;
	}
	auto end = std::chrono::steady_clock::now();
	return end - start;
}

// Represents a named benchmark with a function returning a success flag
struct benchmark_t
{
	const char* name;
	std::function<bool()> func;
};