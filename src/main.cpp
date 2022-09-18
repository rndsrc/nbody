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

int
main(int argc, char *argv[])
{
	(void)printf("nbody: simpmle SYCL tester\n");

	dpc_common::TimeInterval timer;

 	//==============================================================
	// SETUP

	int n = argc <= 1 ? 1024 : atoi(argv[1]);
	(void)printf("%d bodys will be used\n", n);

	// Timesteps depend on each other, so make the queue inorder
	property_list properties{property::queue::in_order()};

	// Define device selector as 'default'
	default_selector device_selector;

	// Create a device queue using DPC++ class queue
	queue q(device_selector, dpc_common::exception_handler, properties);

	// Allocate memory
	real *states = malloc_shared<real>(6 * n, q);

	//==============================================================
	// MAIN LOOP

	for(int i = 0; i < 100; ++i) {
		(void)printf("Step %d: ", i);

		double elapsed = timer.Elapsed();
		(void)printf("%.3g sec\n", elapsed);
	}

 	//==============================================================
	// CLEAN UP

	free(states);

	return 0;
}
