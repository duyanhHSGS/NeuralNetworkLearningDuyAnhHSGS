#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

constexpr uint64_t BIG_NUM = 1000000;

void dummy_work(int i) {
	// Thread-local RNG engine so every thread gets their own state
	static thread_local std::mt19937 rng(std::random_device{}() ^ (i + time(nullptr)));
	std::uniform_int_distribution<int> dist(0, 99);

	volatile double x = i * 1.000001;

	for (int j = 0; j < 1000; ++j) {
		int r = dist(rng);

		if (r < 20) {
			x = std::log1p(std::fabs(x)) * std::tan(x);
		} else if (r < 50) {
			x = std::sin(std::cos(std::sqrt(x * x + j * 0.5)));
		} else if (r < 80) {
			x *= std::pow(x + 1.001, 0.75);
		} else {
			x += std::exp(-x);
		}

		x = std::fmod(x, 3.14159); // keep it bounded
	}

	static volatile double sink;
	sink = x;
}

// 1. SINGLE CORE VERSION
void single_core() {
	for (int i = 0; i < BIG_NUM; ++i)
		dummy_work(i);
}

// 2. OPENMP VERSION
void openmp_version() {
#ifdef _OPENMP
	#pragma omp parallel for
	for (int i = 0; i < BIG_NUM; ++i)
		dummy_work(i);
#else
	std::cout << "OpenMP not supported on this compiler!\n";
#endif
}

// 3. STD::THREAD VERSION
void thread_version() {
	unsigned int n_threads = std::thread::hardware_concurrency();
	if (n_threads == 0) n_threads = 4;
	std::vector<std::thread> threads;

	int chunk_size = BIG_NUM / n_threads;

	for (unsigned int t = 0; t < n_threads; ++t) {
		int start = t * chunk_size;
		int end = (t == n_threads - 1) ? BIG_NUM : (start + chunk_size);
		threads.emplace_back([start, end]() {
			for (int i = start; i < end; ++i)
				dummy_work(i);
		});
	}

	for (auto& th : threads)
		th.join();
}

int main() {
	std::cout << "=== MULTITHREADING BATTLE ROYALE ===\n";

	auto time_it = [](auto&& fn, const char* label) {
		auto start = std::chrono::high_resolution_clock::now();
		fn();
		auto end = std::chrono::high_resolution_clock::now();
		double ms = std::chrono::duration<double, std::milli>(end - start).count();
		std::cout << label << " took: " << ms << " ms\n";
	};

	// time_it(single_core, "Mr. Single");
	time_it(openmp_version, "Mr. OpenMP");
	time_it(thread_version, "Mr. Thread");

	std::cout << "=== BATTLE OVER! WHO WON?! ===\n";
}
