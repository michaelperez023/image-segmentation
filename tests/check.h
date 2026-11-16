// Minimal test harness shared by the test programs.
//
// A test program lists its tests in a table and calls RunTests from main.
// Usage: <program> [name [images_dir]]   runs one test, or all of them
// without a name or with the name "all". images_dir is the repository's
// images/ folder, needed by the tests that read the sample images. The exit
// status is 0 when every check passes.

#ifndef TESTS_CHECK_H
#define TESTS_CHECK_H

#include <cstdio>
#include <cstring>
#include <string>

namespace check
{
	inline int g_failures = 0;
	inline std::string g_images_dir;

	inline void Check(bool ok, const char* expression, const char* file, int line)
	{
		if (!ok)
		{
			std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
			++g_failures;
		}
	}

	struct Test
	{
		const char* name;
		void (*run)();
	};

	template <std::size_t N>
	int RunTests(const Test (&tests)[N], int argc, char** argv)
	{
		const char* only = argc > 1 && std::strcmp(argv[1], "all") != 0 ? argv[1] : nullptr;
		if (argc > 2)
			g_images_dir = argv[2];
		bool found = false;
		for (const Test& test : tests)
		{
			if (only && std::strcmp(only, test.name) != 0)
				continue;
			found = true;
			const int before = g_failures;
			std::printf("[ RUN  ] %s\n", test.name);
			test.run();
			std::printf("[ %s ] %s\n", g_failures == before ? " OK " : "FAIL", test.name);
		}
		if (!found)
		{
			std::fprintf(stderr, "Unknown test: %s\n", only);
			return 1;
		}
		return g_failures == 0 ? 0 : 1;
	}
}

#define CHECK(expression) check::Check((expression), #expression, __FILE__, __LINE__)

#endif // TESTS_CHECK_H
