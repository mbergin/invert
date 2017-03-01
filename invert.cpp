#include <iostream>
#include <chrono>
#include <iomanip>
#include <bitset>
#include <array>
#include <thread>
#include "eigen/Dense"

#include "gray.h"
#include "matrix.h"
#include "benchmark.h"

// A single threaded naive approach that computes the inverse for every combination
bool eigen_random()
{
	const auto size = 7;
	auto success = true;
	for (int i = 0; i < 35*4; ++i)
	{
		auto randomMatrix = Eigen::MatrixXd::Random(size, size);
		Eigen::MatrixXd inverse = randomMatrix.inverse();
		success = success && inverse.allFinite();
	}
	return success;
}

// Directly compute the inverse for every combination, but use OpenMP to make use of 
// the embarrassingly parallel nature of the problem.
bool eigen_random_openmp()
{
	const auto size = 7;
	auto success = true;
	#pragma omp parallel for reduction(&&: success)
	for (int i = 0; i < 35 * 4; ++i)
	{
		auto randomMatrix = Eigen::MatrixXd::Random(size, size);
		Eigen::MatrixXd inverse = randomMatrix.inverse();
		success = success && inverse.allFinite();
	}
	return success;
}

// This approach computes an inverse directly for the initial combination, but then
// uses the Sherman-Morrison formula to update the inverse. The combinations are 
// generated in a way that each differs from the last by the replacement of one item,
// which corresponds to replaceing one row and one column in the combination matrix.
bool eigen_sherman()
{
	auto success = true;

	const auto size = 11;
	const auto select_large = 4;
	const auto size_large = 7;
	const auto select_small = 3;
	const auto size_small = 4;
	const auto comb_size = select_small + select_large;
	
	// Matrix for all items
	Eigen::MatrixXd main = Eigen::MatrixXd::Random(size, size);

	// Generate the first combination
	gray_join_t gray;
	std::bitset<size> selected = gray.next();

	// Matrix for this combination
	Eigen::MatrixXd combination(size_large, size_large);

	// Maps index into main matrix onto an index into the combination matrix
	// and its inverse
	std::array<int, 11> main_to_comb;
	std::array<int, comb_size> comb_to_main;

	auto comb_index = 0;
	for (size_t main_index = 0; main_index < selected.size(); ++main_index)
	{
		if (selected[main_index])
		{
			main_to_comb[main_index] = comb_index;
			comb_to_main[comb_index] = main_index;
			++comb_index;
		}
		else
		{
			main_to_comb[main_index] = -1;
		}
	}

	// Generate the initial combination
	auto ci = 0;
	for (size_t i = 0; i < selected.size(); ++i)
	{
		auto cj = 0;
		for (size_t j = 0; j < selected.size(); ++j)
		{
			if (selected[i] && selected[j]) {
				combination(ci, cj) = main(i, j);
				cj++;
			}
		}
		if (selected[i])
		{
			ci++;
		}
	}

	// Compute the inverse of the initial combination directly
	Eigen::MatrixXd inverse = combination.inverse();
	
	// From now on, update the inverse using two rank-2 updates, to replace a row and column
	for (int n = 1; n < 35*4; ++n)
	{
		// Generate the next combination by removing one item and adding another
		uint32_t selected_next = gray.next();
	
		// Index into main matrix of removed item
		uint32_t removed = set_bit(selected.to_ulong() &~selected_next);

		// Index into main matrix of added item
		uint32_t added = set_bit(selected_next & ~selected.to_ulong());

		// Index into combination matrix of row/column to swap
		auto comb_swap_index = main_to_comb[removed];
	
		// Update the mapping between the main matrix and the combination
		main_to_comb[removed] = -1;
		main_to_comb[added] = comb_swap_index;
		comb_to_main[comb_swap_index] = added;

		// Replacement row and column
		auto new_row = row_map(main, added, comb_to_main);
		auto new_col = col_map(main, added, comb_to_main);

		// Sherman-Morrison u, v vectors for row replacement
		auto u_row = Eigen::VectorXd::Unit(comb_size, comb_swap_index);
		Eigen::RowVectorXd v_row = new_row - combination.row(comb_swap_index);

		// Update the combination matrix and its inverse for the row replacement
		combination.row(comb_swap_index) = new_row;
		inverse = sherman_morrison_update_inverse(inverse, u_row, v_row);

		// Vectors for row replacement
		Eigen::VectorXd u_col = new_col - combination.col(comb_swap_index);
		auto v_col = Eigen::RowVectorXd::Unit(comb_size, comb_swap_index);
		
		// Update the combination matrix and its inverse for the column replacement
		combination.col(comb_swap_index) = new_col;
		inverse = sherman_morrison_update_inverse(inverse, u_col, v_col);

		success = success && inverse.allFinite();

		selected = selected_next;
	}

	return success;
}

int main()
{
	// Benchmark a series of approaches to the problem
	auto iterations = 10000;

	benchmark_t benchmarks[] = {
		{"eigen_random", eigen_random},
		{"eigen_random_openmp", eigen_random_openmp},
		{"eigen_sherman", eigen_sherman},
	};

	for (const auto& benchmark : benchmarks)
	{
		std::cout << std::left << std::setw(30) << benchmark.name;
		std::cout << time_func(benchmark.func, iterations).count() << "s" << std::endl;
	}

	// Processor: Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz, 3901 Mhz, 4 Core(s), 8 Logical Processor(s)
	// Compiler: Visual C++ 2017 RC 64-bit
	// Results for 10000 iterations:
	// eigen_random                  6.23933s
	// eigen_random_openmp           1.58229s
	// eigen_sherman                 3.14105s

	// Possible improvements: 
	//
	// * The Sherman update method is promising in a single threaded context, so
	//   I'd want to split the work across the available hardware threads while still using this method.
	//   Each thread would compute an initial inverse directly to avoid dependencies betweeen threads.
	//   This could end up being faster than eigen_random_openmp.
	//
	// * Try the CUDA version of BLAS for matrix multiplication to parallelise further. It depends whether
	//   the cost of copying between system RAM and GPU RAM is worth the gain in speed.
	//
	// * Investigate block inverse, because the matrix naturally splits into 4x4 and 3x3 diagonal blocks
	//   and these sizes have closed form inverses which might be quicker to compute. One of the four blocks
	//   remains the same between adjacent combinations so its inverse could be cached.
	//
	// * Benchmark different matrix libraries (LAPACK etc).

	return 0;
}
