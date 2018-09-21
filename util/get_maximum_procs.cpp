#include<iostream>
#include<string>
#include<omp.h>

using namespace std;

int main(){
	int numProcs = omp_get_num_procs();
	cout<<numProcs;
	return 0;
}
