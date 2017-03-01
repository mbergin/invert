#pragma once
#include <cstdint>

// count number of bits set in x
inline uint32_t count_bits(uint32_t x)
{
	uint32_t count = 0;
	for (; x; x >>= 1)
	{
		count += x & 1;
	}
	return count;
}

// Index of most significant set bit in x
inline int set_bit(uint32_t x)
{
	uint32_t r = 0;
	while (x >>= 1)
	{
		++r;
	}
	return r;
}

// Factorial of x
inline uint32_t fact(uint32_t x)
{
	uint32_t f = 1;
	for(; x > 1; --x)
	{
		f *= x;
	}
	return f;
}

// Generates Gray Code sequences of length size with pick bits set,
// suitable for combinations. Each successive value has a Hamming
// distance of 2 from the previous value which corresponds to replacing
// one item in a combination with a different item.
class gray_generator_t
{
	int size_;
	int pick_;
	bool reversed_;
	uint32_t index_;
	int combinations_;
public:
	gray_generator_t(int size, int pick)
		: 
	size_(size), 
	pick_(pick), 
	reversed_(false),
	index_(0),
	combinations_(fact(size) / (fact(pick) * fact(size-pick)))
	{
		next();
	}

	// Binary to Gray Code
	static uint32_t gray(uint32_t x)
	{
		return x ^ (x >> 1);
	}

	// Advance to the next Code. At the end, the sequence is replayed in reverse.
	void next()
	{
		auto next_index = index_;
		if (reversed_)
		{
			for (;;)
			{
				--next_index;
				if (next_index == 0)
				{
					reversed_ = false;
					break;
				}
				if (count_bits(gray(next_index)) == pick_)
				{
					index_ = next_index;
					break;
				}
			}
		}
		else
		{
			for (;;)
			{
				++next_index;
				if (next_index == 1 << size_)
				{
					reversed_ = true;
					break;
				}
				if (count_bits(gray(next_index)) == pick_)
				{
					index_ = next_index;
					break;
				}
			}
		}
		
	}

	// The current Gray code
	uint32_t value() const
	{
		return gray(index_);
	}

	// The size of the set being selected from
	int size() const
	{
		return size_;
	}

	// Length of the sequence that this will generate
	int combinations() const
	{
		return combinations_;
	}

};

// Combines two Gray code combinatorial generators such that only one item
// is replaced from one selection to the next.
class gray_join_t
{
	gray_generator_t small_{4, 3};
	gray_generator_t large_{7, 4};
	int count_ = 0;
public:
	uint32_t next()
	{
		uint32_t ret = small_.value() << large_.size() | large_.value();
		count_++;
		if (count_ > 0 && count_ % large_.combinations() == 0)
		{
			small_.next();
		}	
		large_.next();
		return ret;
	}

};
