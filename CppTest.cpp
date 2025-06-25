#include<bits\stdc++.h>

template<int N>
struct Hello {
	Hello() {
		std::cout << "I was compiled for N = " << N << "\n";
	}
};

int main() {
	Hello<3> a;
	Hello<5> b;
}
