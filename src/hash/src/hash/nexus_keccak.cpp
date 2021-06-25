#include "hash/nexus_keccak.hpp"

NexusKeccak::NexusKeccak()
{
}

NexusKeccak::NexusKeccak(const Int_array<uint64_t, 16>& m) 
{
	setMessage(m);
}

void NexusKeccak::setMessage(const Int_array<uint64_t, 16>& m)
{
	//convert the input array into two message arrays in preparation for absorption
	for (int i = 0; i < messageLength; i++)
	{
		message1[i] = m[i];
	}
	for (int i = 0; i < 7; i++)
	{
		message2[i] = m[i + messageLength];
	}
	message2[7] = NXS_SUFFIX_1;
	message2[8] = NXS_SUFFIX_2;

}

NexusKeccak::k_state NexusKeccak::messageToState(const k_message& m)
{
	k_state s;
	for (auto i = 0; i < messageLength; i++)
		s[i / 5][i % 5] = m[i];

	return s;
}

NexusKeccak::k_state NexusKeccak::state_XOR(const k_state& s, const k_message& m)
//xor a state and a message
{
	k_state result = s;
	for (int i = 0; i < messageLength; i++)
	{
		result[i / 5][i % 5] = s[i / 5][i % 5] ^ m[i];
	}
	return result;
}

NexusKeccak::k_state NexusKeccak::theta(const k_state& s)
{
	return k_state();
}

NexusKeccak::k_state NexusKeccak::keccak_round(const k_state& state_in, int round)
{

	k_state A = state_in;
	//temp variables
	k_state B;
	k_plane C, D;

	//theta
	for (auto x = 0; x < 5; x++)
		C[x] = A[0][x] ^ A[1][x] ^ A[2][x] ^ A[3][x] ^ A[4][x];	
	
	for (auto x = 0; x < 5; x++)
		D[x] = C[(x+4) % 5] ^ rol(C[(x + 1) % 5], 1);

	for (auto x = 0; x < 5; x++)
		for (auto y = 0; y < 5; y++)
			A[y][x] = A[y][x] ^ D[x];
	
	//rho and pi
	for (auto x = 0; x < 5; x++)
		for (auto y = 0; y < 5; y++)
			B[y][(2 * x + 3 * y) % 5] = rol(A[y][x], r[x][y]);

	//chi
	for (auto x = 0; x < 5; x++)
		for (auto y = 0; y < 5; y++)
			A[y][x] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]);

	//Iota step
	A[0][0] = A[0][0] ^ round_const[round];

	return A;
}

void NexusKeccak::calculateHash()
{
	//start with the first part of the message
	k_state s = messageToState(message1);

	//1st set of 24 rounds of keccak
	for (int i = 0; i < numRounds; i++)
	{
		s = keccak_round(s, i);
	}

	//xor the state with the next part of the message
	s = state_XOR(s, message2);
	//2nd set of 24 rounds of keccak
	for (int i = 0; i < numRounds; i++)
	{
		s = keccak_round(s, i);
	}

	//3rd set of 24 rounds of keccak
	for (int i = 0; i < numRounds; i++)
	{
		s = keccak_round(s, i);
	}
	hash = s;
}

uint64_t NexusKeccak::getResult()
{
	//we really only care about the most siginificant bits. return the top 64 bits only.
	return hash[1][1];
}


