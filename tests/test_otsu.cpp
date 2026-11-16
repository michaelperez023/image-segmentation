// Tests for Otsu's method (Otsu.h): synthetic images, a brute-force search
// over every set of thresholds on random histograms, the sample images, and
// OpenCV's own two-class Otsu threshold.
//
// Usage: otsu_tests [name [images_dir]]   (see check.h)

#include "Otsu.h"
#include "check.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
	using otsu::Histogram;

	bool Near(double a, double b, double relative = 1e-9)
	{
		return std::abs(a - b) <= relative * std::max({1.0, std::abs(a), std::abs(b)});
	}

	template <typename Function>
	bool Throws(Function function)
	{
		try
		{
			function();
		}
		catch (const std::invalid_argument&)
		{
			return true;
		}
		return false;
	}

	std::string Text(const std::vector<int>& values)
	{
		std::string text;
		for (std::size_t i = 0; i < values.size(); ++i)
			text += (i ? "," : "") + std::to_string(values[i]);
		return text;
	}

	// Between-class variance written out from the definition,
	// sum_k w_k (mu_k - mu)^2, summing each class straight from the
	// histogram. Class k holds the levels t[k-1] + 1 to t[k], the last class
	// the levels above t[K-2].
	double DirectBetweenVariance(const Histogram& h, const std::vector<int>& t)
	{
		double n = 0.0, s = 0.0;
		for (std::size_t i = 0; i < h.size(); ++i)
		{
			n += h[i];
			s += static_cast<double>(i) * h[i];
		}
		const double mu = s / n;
		double variance = 0.0;
		int first = 0;
		for (std::size_t k = 0; k <= t.size(); ++k)
		{
			const int last = k < t.size() ? t[k] : static_cast<int>(h.size()) - 1;
			double nk = 0.0, sk = 0.0;
			for (int i = first; i <= last; ++i)
			{
				nk += h[i];
				sk += i * h[i];
			}
			if (nk > 0.0)
				variance += nk / n * (sk / nk - mu) * (sk / nk - mu);
			first = last + 1;
		}
		return variance;
	}

	// Calls `visit` with every increasing set of `count` thresholds from 0 to
	// levels - 2, in lexicographic order
	void ForEachThresholds(int levels, int count, const std::function<void(const std::vector<int>&)>& visit)
	{
		std::vector<int> t(count);
		for (int k = 0; k < count; ++k)
			t[k] = k;
		while (true)
		{
			visit(t);
			int k = count - 1;
			while (k >= 0 && t[k] == levels - 2 - (count - 1 - k))
				--k;
			if (k < 0)
				return;
			++t[k];
			for (int j = k + 1; j < count; ++j)
				t[j] = t[j - 1] + 1;
		}
	}

	// Every optimum of a brute-force search, in lexicographic order
	struct Optima
	{
		double best = -1.0;
		std::vector<std::vector<int>> thresholds;
	};

	Optima BruteForce(const Histogram& h, int classes)
	{
		Optima optima;
		ForEachThresholds(static_cast<int>(h.size()), classes - 1,
		                  [&](const std::vector<int>& t) { optima.best = std::max(optima.best, DirectBetweenVariance(h, t)); });
		ForEachThresholds(static_cast<int>(h.size()), classes - 1, [&](const std::vector<int>& t) {
			if (Near(DirectBetweenVariance(h, t), optima.best, 1e-12))
				optima.thresholds.push_back(t);
		});
		return optima;
	}

	// True if two sets of thresholds split the histogram into the same
	// classes, that is, no pixel lies between t[k] and u[k]
	bool SameClasses(const Histogram& h, const std::vector<int>& t, const std::vector<int>& u)
	{
		if (t.size() != u.size())
			return false;
		for (std::size_t k = 0; k < t.size(); ++k)
			for (int i = std::min(t[k], u[k]) + 1; i <= std::max(t[k], u[k]); ++i)
				if (h[i] > 0.0)
					return false;
		return true;
	}

	// Random histograms of several shapes; `shape` 0: every level occupied,
	// 1: most levels empty, 2: a mixture of peaks, 3: only a few levels
	// occupied (many equal scores).
	Histogram RandomHistogram(cv::RNG& rng, int levels, int shape)
	{
		Histogram h(levels, 0.0);
		switch (shape)
		{
		case 0:
			for (double& count : h)
				count = rng.uniform(1, 1000);
			break;
		case 1:
			for (double& count : h)
				count = rng.uniform(0.0, 1.0) < 0.7 ? 0.0 : rng.uniform(1, 1000);
			break;
		case 2:
		{
			const int peaks = rng.uniform(2, 5);
			for (int p = 0; p < peaks; ++p)
			{
				const double centre = rng.uniform(0.0, static_cast<double>(levels));
				const double width = rng.uniform(0.5, std::max(1.0, levels / 8.0));
				const double height = rng.uniform(100.0, 5000.0);
				for (int i = 0; i < levels; ++i)
					h[i] += std::floor(height * std::exp(-0.5 * (i - centre) * (i - centre) / (width * width)));
			}
			break;
		}
		default:
		{
			const int occupied = rng.uniform(1, 6);
			for (int k = 0; k < occupied; ++k)
				h[rng.uniform(0, levels)] += rng.uniform(1, 1000);
			break;
		}
		}
		if (std::all_of(h.begin(), h.end(), [](double count) { return count == 0.0; }))
			h[rng.uniform(0, levels)] = 1.0;
		return h;
	}

	cv::Mat ReadSample(const std::string& name, int flags = cv::IMREAD_COLOR)
	{
		const std::string path = check::g_images_dir + "/" + name;
		const cv::Mat image = cv::imread(path, flags);
		if (image.empty())
			std::fprintf(stderr, "could not read %s (pass the images directory as the second argument)\n", path.c_str());
		CHECK(!image.empty());
		return image;
	}

	// Four vertical bands at grey levels 30, 90, 150 and 210, of different
	// widths, with uniform noise from -12 to 12, so the levels of the bands
	// are 18-42, 78-102, 138-162 and 198-222.
	void TestFourLevels()
	{
		const int widths[] = {20, 40, 60, 40};
		const int levels[] = {30, 90, 150, 210};
		cv::Mat image(120, 160, CV_8U), band(image.size(), CV_8U);
		cv::RNG rng(4410);
		for (int r = 0; r < image.rows; ++r)
		{
			int c = 0;
			for (int b = 0; b < 4; ++b)
			{
				for (int i = 0; i < widths[b]; ++i, ++c)
				{
					image.at<uchar>(r, c) = static_cast<uchar>(levels[b] + rng.uniform(-12, 13));
					band.at<uchar>(r, c) = static_cast<uchar>(b);
				}
			}
		}

		const otsu::Result result = otsu::FindThresholds(image, 4);
		std::printf("thresholds %s, eta %.4f\n", Text(result.thresholds).c_str(), result.Eta());
		CHECK(result.thresholds.size() == 3);
		if (result.thresholds.size() != 3)
			return;

		// Each threshold is the brightest level of its band (the lowest of the
		// equally good thresholds in the empty gap above it)
		for (int b = 0; b < 3; ++b)
		{
			double highest = 0.0;
			cv::minMaxLoc(image, nullptr, &highest, nullptr, nullptr, band == b);
			CHECK(result.thresholds[b] == static_cast<int>(highest));
		}
		CHECK(cv::countNonZero(otsu::ClassIndices(image, result.thresholds) != band) == 0);
		for (int b = 0; b < 4; ++b)
			CHECK(result.class_sizes[b] == 120.0 * widths[b]);
		CHECK(result.Eta() > 0.98 && result.Eta() <= 1.0);

		const cv::Mat classes = otsu::ClassImage(image, result.thresholds);
		const uchar grey[] = {0, 85, 170, 255};
		for (int b = 0; b < 4; ++b)
			CHECK(cv::countNonZero((classes == grey[b]) != (band == b)) == 0);
	}

	// Every set of thresholds is scored with the textbook formula and the
	// best is compared with FindThresholds
	void TestBruteForce()
	{
		struct Case
		{
			int levels, classes, histograms;
		};
		const Case cases[] = {
			{2, 2, 4}, {3, 3, 4}, {5, 4, 8}, {8, 2, 8}, {8, 3, 8}, {8, 4, 8}, {16, 4, 8},
			{40, 3, 8}, {64, 4, 8}, {256, 2, 8}, {256, 3, 4},
		};
		cv::RNG rng(20191029);
		int compared = 0, with_ties = 0;
		for (const Case& c : cases)
		{
			for (int n = 0; n < c.histograms; ++n)
			{
				const Histogram h = RandomHistogram(rng, c.levels, n % 4);
				const otsu::Result result = otsu::FindThresholds(h, c.classes);
				const Optima optima = BruteForce(h, c.classes);
				++compared;
				with_ties += optima.thresholds.size() > 1;

				const bool optimal =
					std::find(optima.thresholds.begin(), optima.thresholds.end(), result.thresholds) !=
					optima.thresholds.end();
				CHECK(optimal);
				CHECK(Near(result.between_variance, optima.best));
				CHECK(Near(otsu::BetweenClassVariance(h, result.thresholds), optima.best));
				// When all optima give the same classes, the lowest is returned
				bool one_partition = true;
				for (const std::vector<int>& t : optima.thresholds)
					one_partition = one_partition && SameClasses(h, t, optima.thresholds.front());
				if (one_partition)
					CHECK(result.thresholds == optima.thresholds.front());
				if (!optimal)
					std::fprintf(stderr, "levels %d, classes %d: got %s, best %s\n", c.levels, c.classes,
					             Text(result.thresholds).c_str(), Text(optima.thresholds.front()).c_str());
			}
		}

		// Four classes of 256 levels: all 2.7 million sets, scored from
		// cumulative class weights and means
		for (int n = 0; n < 3; ++n)
		{
			const Histogram h = RandomHistogram(rng, 256, n);
			std::vector<double> w(257, 0.0), m(257, 0.0);
			for (int i = 0; i < 256; ++i)
			{
				w[i + 1] = w[i] + h[i];
				m[i + 1] = m[i] + i * h[i];
			}
			const double total = w[256], mu = m[256] / total;
			auto term = [&](int first, int last) {
				const double weight = (w[last + 1] - w[first]) / total;
				if (weight <= 0.0)
					return 0.0;
				const double mean = (m[last + 1] - m[first]) / (w[last + 1] - w[first]);
				return weight * (mean - mu) * (mean - mu);
			};
			double best = -1.0;
			for (int t1 = 0; t1 <= 252; ++t1)
				for (int t2 = t1 + 1; t2 <= 253; ++t2)
					for (int t3 = t2 + 1; t3 <= 254; ++t3)
						best = std::max(best, term(0, t1) + term(t1 + 1, t2) + term(t2 + 1, t3) + term(t3 + 1, 255));
			const otsu::Result result = otsu::FindThresholds(h, 4);
			CHECK(Near(result.between_variance, best));
			CHECK(Near(DirectBetweenVariance(h, result.thresholds), best));
			++compared;
		}
		std::printf("%d histograms compared, %d of them with several optimal threshold sets\n", compared, with_ties);
	}

	// Thresholds of the sample images, from a separate exhaustive search in
	// Python (NumPy). scikit-image's threshold_multiotsu (checked with 0.22
	// and 0.26) gives the same thresholds for all but edge_L, where it
	// returns 172, 189, 195, whose classes score lower. Its histogram of
	// edge_L has one bin per level from 171 to 196, and its search starts the
	// cumulative first moment at the first bin's weight instead of 0: when
	// level 171 (15% of the pixels) shares the lowest class with brighter
	// levels, its pixels count as if they were at 172.
	struct Sample
	{
		const char* file;
		int threshold2;
		std::vector<int> thresholds4;
		double eta4;
	};

	const Sample kSamples[] = {
		{"meanshift/S00-150x150.png", 79, {56, 91, 144}, 0.8683},
		{"meanshift/S02-150x150.png", 161, {65, 103, 183}, 0.9634},
		{"meanshift/S03-150x150.png", 134, {52, 94, 165}, 0.9432},
		{"meanshift/set1Seq2_L-150x150.png", 144, {84, 141, 187}, 0.9400},
		{"meanshift/set2Seq1_L-150x150.png", 154, {87, 147, 184}, 0.8968},
		{"otsu2/andreas_L-150x150.png", 67, {39, 75, 119}, 0.9446},
		{"otsu2/edge_L-150x150.png", 172, {171, 172, 189}, 0.9997},
		{"otsu2/south_L-150x150.png", 132, {75, 152, 214}, 0.9805},
		{"otsu4/S01-150x150.png", 101, {66, 103, 138}, 0.8786},
		{"otsu4/S04-150x150.png", 137, {59, 102, 169}, 0.9507},
		{"otsu4/S05-150x150.png", 116, {71, 115, 168}, 0.8898},
		{"otsu4/blocks_L-150x150.png", 109, {72, 114, 159}, 0.9090},
	};

	void TestSampleImages()
	{
		for (const Sample& sample : kSamples)
		{
			const cv::Mat image = ReadSample(sample.file);
			if (image.empty())
				continue;
			const cv::Mat gray = otsu::ToGray(image);
			const Histogram h = otsu::ComputeHistogram(gray);
			const otsu::Result two = otsu::FindThresholds(h, 2);
			const otsu::Result four = otsu::FindThresholds(h, 4);
			std::printf("%-34s 2 classes: %3d   4 classes: %-11s eta %.4f\n", sample.file, two.thresholds[0],
			            Text(four.thresholds).c_str(), four.Eta());
			CHECK(two.thresholds == std::vector<int>{sample.threshold2});
			CHECK(four.thresholds == sample.thresholds4);
			CHECK(SameClasses(h, four.thresholds, sample.thresholds4));
			CHECK(std::abs(four.Eta() - sample.eta4) < 6e-5);
			// The brightest class is not empty (the 2019 search left it with
			// 3 to 53 pixels on the four images meant for this method)
			CHECK(four.class_sizes[3] > 100.0);
		}
	}

	// The 2019 code kept its histogram in a global that was never cleared,
	// so a second call on the same image gave other thresholds.
	void TestRepeatCalls()
	{
		cv::RNG rng(7);
		cv::Mat a(90, 110, CV_8U), b(60, 80, CV_8U);
		rng.fill(a, cv::RNG::NORMAL, 100, 40);
		rng.fill(b, cv::RNG::UNIFORM, 0, 256);
		const otsu::Result first = otsu::FindThresholds(a, 4);
		const otsu::Result second = otsu::FindThresholds(a, 4);
		otsu::FindThresholds(b, 4);
		const otsu::Result third = otsu::FindThresholds(a, 4);
		CHECK(first.thresholds == second.thresholds && first.thresholds == third.thresholds);
		CHECK(first.between_variance == second.between_variance && first.between_variance == third.between_variance);

		const Histogram h1 = otsu::ComputeHistogram(a);
		const Histogram h2 = otsu::ComputeHistogram(a);
		CHECK(h1 == h2);
		double total = 0.0;
		for (double count : h1)
			total += count;
		CHECK(total == static_cast<double>(a.total()));
	}

	// Two classes must split each image exactly as cv::threshold with
	// THRESH_OTSU does
	void TestOpenCvOtsu()
	{
		std::vector<cv::Mat> images;
		cv::RNG rng(1979);
		for (int n = 0; n < 6; ++n)
		{
			cv::Mat image(50 + 7 * n, 70 - 5 * n, CV_8U);
			if (n % 3 == 0)
				rng.fill(image, cv::RNG::UNIFORM, 0, 256);
			else if (n % 3 == 1)
				rng.fill(image, cv::RNG::NORMAL, 60 + 20 * n, 25);
			else
			{
				// Two populations of different sizes
				rng.fill(image, cv::RNG::NORMAL, 70, 15);
				cv::Mat bright(image.size(), CV_8U);
				rng.fill(bright, cv::RNG::NORMAL, 180, 20);
				bright(cv::Rect(0, 0, image.cols / 3, image.rows)).copyTo(image(cv::Rect(0, 0, image.cols / 3, image.rows)));
			}
			images.push_back(image);
		}
		if (!check::g_images_dir.empty())
		{
			for (const Sample& sample : kSamples)
			{
				const cv::Mat image = ReadSample(sample.file);
				if (image.empty())
					continue;
				images.push_back(otsu::ToGray(image));
				std::vector<cv::Mat> channels;
				cv::split(image, channels);
				images.insert(images.end(), channels.begin(), channels.end());
			}
		}

		int same = 0;
		for (const cv::Mat& image : images)
		{
			const otsu::Result result = otsu::FindThresholds(image, 2);
			cv::Mat expected;
			const double opencv_threshold = cv::threshold(image, expected, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
			const cv::Mat binary = otsu::ClassImage(image, result.thresholds);
			const bool equal = cv::countNonZero(binary != expected) == 0;
			CHECK(equal);
			if (!equal)
				std::fprintf(stderr, "threshold %d, cv::threshold %g\n", result.thresholds[0], opencv_threshold);
			same += equal;
		}
		std::printf("%d of %zu images split as by cv::threshold\n", same, images.size());
	}

	// Pixels equal to a threshold go to the lower class, in the search and
	// in the class image alike (the 2019 code searched with t in the lower
	// class but drew it in the upper one)
	void TestClassBoundaries()
	{
		cv::Mat ramp(16, 16, CV_8U);
		for (int v = 0; v < 256; ++v)
			ramp.at<uchar>(v / 16, v % 16) = static_cast<uchar>(v);
		const std::vector<int> t = {50, 100, 200};
		const cv::Mat indices = otsu::ClassIndices(ramp, t);
		for (int v = 0; v < 256; ++v)
		{
			const int expected = v <= 50 ? 0 : v <= 100 ? 1 : v <= 200 ? 2 : 3;
			CHECK(indices.at<uchar>(v / 16, v % 16) == expected);
		}

		// edge_L has only six grey levels, so its thresholds sit on occupied
		// levels: the four classes are {171}, {172}, {188, 189}, {195, 196}
		Histogram h(256, 0.0);
		h[171] = 3454;
		h[172] = 6296;
		h[188] = 145;
		h[189] = 5;
		h[195] = 918;
		h[196] = 11682;
		const otsu::Result result = otsu::FindThresholds(h, 4);
		CHECK(result.thresholds == (std::vector<int>{171, 172, 189}));
		CHECK(result.class_sizes == (std::vector<double>{3454, 6296, 150, 12600}));

		// The class sizes and sigma_B^2 match the pixels of the class image
		cv::Mat image(100, 120, CV_8U);
		cv::RNG rng(31);
		rng.fill(image, cv::RNG::NORMAL, 128, 50);
		const otsu::Result random = otsu::FindThresholds(image, 4);
		const cv::Mat classes = otsu::ClassIndices(image, random.thresholds);
		cv::Mat as_double;
		image.convertTo(as_double, CV_64F);
		const double mean = cv::mean(as_double)[0];
		double between = 0.0;
		for (int k = 0; k < 4; ++k)
		{
			const cv::Mat mask = classes == k;
			const int count = cv::countNonZero(mask);
			CHECK(count == static_cast<int>(random.class_sizes[k]));
			if (count > 0)
			{
				const double class_mean = cv::mean(as_double, mask)[0];
				between += count * (class_mean - mean) * (class_mean - mean);
			}
		}
		between /= static_cast<double>(image.total());
		CHECK(Near(between, random.between_variance, 1e-9));
	}

	// The 2019 method for colour images: two classes per channel
	void TestPerChannel()
	{
		cv::Mat colour(40, 60, CV_8UC3);
		cv::RNG rng(11);
		rng.fill(colour, cv::RNG::UNIFORM, cv::Scalar(0, 30, 60), cv::Scalar(200, 230, 256));
		std::vector<cv::Mat> images = {colour};
		if (!check::g_images_dir.empty())
		{
			images.push_back(ReadSample("otsu2/andreas_L-150x150.png"));
			images.push_back(ReadSample("otsu4/blocks_L-150x150.png"));
		}
		for (const cv::Mat& image : images)
		{
			if (image.empty())
				continue;
			std::vector<otsu::Result> results;
			const cv::Mat binary = otsu::ThresholdChannels(image, &results);
			CHECK(results.size() == 3);
			std::vector<cv::Mat> channels, expected(3);
			cv::split(image, channels);
			for (int c = 0; c < 3; ++c)
				cv::threshold(channels[c], expected[c], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
			cv::Mat merged;
			cv::merge(expected, merged);
			CHECK(binary.type() == CV_8UC3);
			CHECK(cv::norm(binary, merged, cv::NORM_INF) == 0.0);
			CHECK(!otsu::IsGray(image));
		}

		// Grey images stored with three equal channels
		cv::Mat gray(30, 30, CV_8U), gray3;
		rng.fill(gray, cv::RNG::UNIFORM, 0, 256);
		cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);
		CHECK(otsu::IsGray(gray) && otsu::IsGray(gray3));
		CHECK(cv::norm(otsu::ToGray(gray3), gray, cv::NORM_INF) == 0.0);
	}

	void TestDegenerate()
	{
		// A constant image: any thresholds are equally good, eta is 0
		const cv::Mat flat(20, 30, CV_8U, cv::Scalar(77));
		for (int classes : {2, 4})
		{
			const otsu::Result result = otsu::FindThresholds(flat, classes);
			CHECK(static_cast<int>(result.thresholds.size()) == classes - 1);
			CHECK(std::is_sorted(result.thresholds.begin(), result.thresholds.end()));
			CHECK(result.Eta() == 0.0);
			CHECK(cv::countNonZero(otsu::ClassIndices(flat, result.thresholds) != otsu::ClassIndices(flat, result.thresholds).at<uchar>(0, 0)) == 0);
		}

		// Two grey levels and four classes: the levels are separated, eta is 1
		cv::Mat two(10, 10, CV_8U, cv::Scalar(10));
		two(cv::Rect(0, 0, 3, 10)).setTo(240);
		const otsu::Result result = otsu::FindThresholds(two, 4);
		const cv::Mat indices = otsu::ClassIndices(two, result.thresholds);
		CHECK(indices.at<uchar>(0, 0) > indices.at<uchar>(0, 9));
		CHECK(Near(result.Eta(), 1.0, 1e-12));

		// One pixel, and a histogram with as many levels as classes
		CHECK(otsu::FindThresholds(cv::Mat(1, 1, CV_8U, cv::Scalar(5)), 4).thresholds.size() == 3);
		CHECK(otsu::FindThresholds(Histogram{1, 2, 3, 4}, 4).thresholds == (std::vector<int>{0, 1, 2}));

		// Invalid input
		CHECK(Throws([&] { otsu::FindThresholds(flat, 1); }));
		CHECK(Throws([&] { otsu::FindThresholds(flat, 257); }));
		CHECK(Throws([] { otsu::FindThresholds(Histogram(256, 0.0), 2); }));
		CHECK(Throws([] { otsu::FindThresholds(Histogram{1, -1, 3}, 2); }));
		CHECK(Throws([] { otsu::ComputeHistogram(cv::Mat()); }));
		CHECK(Throws([] { otsu::ComputeHistogram(cv::Mat(4, 4, CV_8UC3, cv::Scalar::all(0))); }));
		CHECK(Throws([] { otsu::ComputeHistogram(cv::Mat(4, 4, CV_16U, cv::Scalar(0))); }));
		CHECK(Throws([&] { otsu::ClassIndices(flat, {}); }));
		CHECK(Throws([&] { otsu::ClassIndices(flat, {5, 5}); }));
		CHECK(Throws([&] { otsu::ClassImage(flat, {255}); }));
		CHECK(Throws([] { otsu::ToGray(cv::Mat(4, 4, CV_32F, cv::Scalar(0))); }));
	}

	const check::Test kTests[] = {
		{"four_levels", TestFourLevels},
		{"brute_force", TestBruteForce},
		{"sample_images", TestSampleImages},
		{"repeat_calls", TestRepeatCalls},
		{"opencv_otsu", TestOpenCvOtsu},
		{"class_boundaries", TestClassBoundaries},
		{"per_channel", TestPerChannel},
		{"degenerate", TestDegenerate},
	};
}

int main(int argc, char** argv)
{
	return check::RunTests(kTests, argc, argv);
}
