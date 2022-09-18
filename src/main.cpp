// Copyright (C) 2022 Chi-kwan Chan
// Copyright (C) 2022 Steward Observatory
//
// This file is part of nbody.
//
// nbody is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// nbody is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with nbody.  If not, see <http://www.gnu.org/licenses/>.

#include "nbody.hpp"

#include <cstdio>
#include <cstdlib>

#include <CL/sycl.hpp>

#include "dpc_common.hpp"

using namespace sycl;

void
output(int i, int N, real *s)
{
	char fname[256];
	sprintf(fname, "%06d.raw", i);

	FILE *file = fopen(fname, "wb");
	fwrite(s, sizeof(real), N, file);
	fclose(file);
}

int
main(int argc, char *argv[])
{
	(void)printf("nbody: simpmle SYCL tester\n");

	dpc_common::TimeInterval timer;

 	//==============================================================
	// INSTANTIZATION

	int n = argc <= 1 ? 256 : atoi(argv[1]); // number of particles
	int t = argc <= 2 ? 32  : atoi(argv[2]); // number of outer time loop
	int s = argc <= 2 ? 128 : atoi(argv[2]); // number of inner time loop
	(void)printf("Configurations:\t%d-body with %d x %d steps\n", n, t, s);

	// Timesteps depend on each other, so make the queue inorder
	property_list properties{property::queue::in_order()};

	// Define device selector as 'default'
	default_selector device_selector;

	// Create a device queue using DPC++ class queue
	queue q(device_selector, dpc_common::exception_handler, properties);

	// Allocate memory
	const int o = 2;         // order of ODEs
	const int d = 3;         // spacial dimensions
	const int N = o * d * n; // number of reals to describe the states
	real *states = malloc_shared<real>(N, q);

	(void)printf("Instantized:\t%.3g sec\n", timer.Elapsed());

	//==============================================================
	// INITIALIZATION

	// Fill positions and velocities with random values in [-1,1]
	for(int i = 0; i < N; ++i)
		states[i] = 2.0 * rand() / RAND_MAX - 1.0;
	output(0, N, states);

	(void)printf("Initialized:\t%.3g sec\n", timer.Elapsed());

	//==============================================================
	// MAIN LOOP

	const real f = pow(2.0, 1.0/s);

	for(int i = 0; i < t; ++i) {
		(void)printf("%6d:\t", i);

		// Submit the same kernel s times for the substeps
		for(int j = 0; j < s; ++j)
			q.submit([&](handler& h) {
				h.parallel_for(range<1>(N), [=] (id<1> i) {
					states[i] *= f;
				});
			});

		q.wait_and_throw();
		double ct = timer.Elapsed();

		output(i+1, N, states);
		double io = timer.Elapsed();

		(void)printf("compute: %.3g sec; I/O: %.3g sec\n", ct, io);
	}

 	//==============================================================
	// CLEAN UP

	free(states);

	return 0;
}
