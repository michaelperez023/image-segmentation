// Tests for mean shift filtering and segmentation (MeanShift.h) on synthetic
// and sample images.
//
// Usage: meanshift_tests [name [images_dir]]   (see check.h)

#include "MeanShift.h"
#include "check.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
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

	double Distance(const cv::Vec3f& p, const cv::Vec3f& q)
	{
		const cv::Vec3d d = cv::Vec3d(p) - cv::Vec3d(q);
		return std::sqrt(d.dot(d));
	}

	cv::Vec3f LabOf(const cv::Vec3b& bgr)
	{
		return BgrToLab(cv::Mat(1, 1, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2]))).at<cv::Vec3f>(0, 0);
	}

	// Four quadrants of the given BGR colours, with Gaussian noise of
	// standard deviation `noise` on each channel. `quadrant` receives the
	// quadrant of each pixel (0 top left, 1 top right, 2 bottom left, 3
	// bottom right).
	cv::Mat Quadrants(cv::Size size, const cv::Vec3b (&colours)[4], double noise, std::uint64_t seed,
	                  cv::Mat* quadrant = nullptr)
	{
		cv::Mat image(size, CV_8UC3), index(size, CV_32S);
		cv::RNG rng(seed);
		for (int r = 0; r < size.height; ++r)
		{
			for (int c = 0; c < size.width; ++c)
			{
				const int q = (r >= size.height / 2 ? 2 : 0) + (c >= size.width / 2 ? 1 : 0);
				index.at<int>(r, c) = q;
				for (int ch = 0; ch < 3; ++ch)
					image.at<cv::Vec3b>(r, c)[ch] = cv::saturate_cast<uchar>(colours[q][ch] + rng.gaussian(noise));
			}
		}
		if (quadrant)
			*quadrant = index;
		return image;
	}

	// Smooth random colours with a few shapes and noise, for the tests that
	// need some structure
	cv::Mat Scene(cv::Size size, std::uint64_t seed)
	{
		cv::RNG rng(seed);
		cv::Mat small(size.height / 8 + 2, size.width / 8 + 2, CV_8UC3), image;
		rng.fill(small, cv::RNG::UNIFORM, 0, 256);
		cv::resize(small, image, size, 0, 0, cv::INTER_CUBIC);
		cv::circle(image, cv::Point(size.width / 3, size.height / 2), size.height / 4, cv::Scalar(30, 160, 230), cv::FILLED);
		cv::rectangle(image, cv::Rect(size.width / 2, size.height / 5, size.width / 4, size.height / 3),
		              cv::Scalar(200, 70, 40), cv::FILLED);
		cv::Mat noise(size, CV_8UC3);
		rng.fill(noise, cv::RNG::NORMAL, 0, 4);
		cv::add(image, noise, image);
		return image;
	}

	cv::Mat ReadSample(const std::string& name)
	{
		const std::string path = check::g_images_dir + "/" + name;
		const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
		if (image.empty())
			std::fprintf(stderr, "could not read %s (pass the images directory as the second argument)\n", path.c_str());
		CHECK(!image.empty());
		return image;
	}

	// The labels form regions 0..Count()-1 with the listed sizes, and each
	// region's colour is the mean filtered colour of its pixels
	void CheckRegions(const MeanShift::Regions& regions, const cv::Mat& filtered)
	{
		const int count = regions.Count();
		CHECK(count >= 1 && static_cast<int>(regions.colours.size()) == count);
		CHECK(regions.labels.type() == CV_32S && regions.labels.size() == filtered.size());
		std::vector<int> sizes(count, 0);
		std::vector<cv::Vec3d> sums(count);
		bool in_range = true;
		for (int r = 0; r < filtered.rows; ++r)
		{
			for (int c = 0; c < filtered.cols; ++c)
			{
				const int label = regions.labels.at<int>(r, c);
				if (label < 0 || label >= count)
				{
					in_range = false;
					continue;
				}
				++sizes[label];
				sums[label] += cv::Vec3d(filtered.at<cv::Vec3f>(r, c));
			}
		}
		CHECK(in_range);
		CHECK(sizes == regions.sizes);
		for (int k = 0; k < count && in_range; ++k)
			CHECK(sizes[k] > 0 && Distance(cv::Vec3f(sums[k] / sizes[k]), regions.colours[k]) < 1e-3);
	}

	// Pure blue must have b* of about -108 (it would be +67, the b* of red,
	// with R and B swapped), and 8-bit colours survive the round trip
	void TestLabConversion()
	{
		struct Known
		{
			cv::Vec3b bgr;
			cv::Vec3f lab;
		};
		const Known known[] = {
			{{255, 0, 0}, {32.30f, 79.19f, -107.86f}},   // blue
			{{0, 0, 255}, {53.24f, 80.09f, 67.20f}},     // red
			{{0, 255, 0}, {87.74f, -86.18f, 83.18f}},    // green
			{{255, 255, 255}, {100.0f, 0.0f, 0.0f}},
			{{0, 0, 0}, {0.0f, 0.0f, 0.0f}},
		};
		for (const Known& k : known)
		{
			const cv::Vec3f lab = LabOf(k.bgr);
			std::printf("BGR (%3d, %3d, %3d) -> L*a*b* (%7.2f, %7.2f, %7.2f)\n", k.bgr[0], k.bgr[1], k.bgr[2], lab[0],
			            lab[1], lab[2]);
			CHECK(Distance(lab, k.lab) < 0.05);
		}

		cv::Mat image(64, 64, CV_8UC3);
		cv::RNG rng(2002);
		rng.fill(image, cv::RNG::UNIFORM, 0, 256);
		const cv::Mat lab = BgrToLab(image);
		CHECK(lab.type() == CV_32FC3);
		CHECK(cv::norm(LabToBgr(lab), image, cv::NORM_INF) <= 1.0);

		// Grey and BGRA images give the same L*a*b* as their BGR versions
		cv::Mat gray(16, 16, CV_8U), gray_bgr, bgra;
		rng.fill(gray, cv::RNG::UNIFORM, 0, 256);
		cv::cvtColor(gray, gray_bgr, cv::COLOR_GRAY2BGR);
		cv::cvtColor(image, bgra, cv::COLOR_BGR2BGRA);
		CHECK(cv::norm(BgrToLab(gray), BgrToLab(gray_bgr), cv::NORM_INF) == 0.0);
		CHECK(cv::norm(BgrToLab(bgra), lab, cv::NORM_INF) == 0.0);
	}

	// Four noisy flat colours: one region per quadrant, with the right colours
	void TestPiecewiseConstant()
	{
		const cv::Vec3b colours[4] = {{40, 40, 200}, {200, 60, 40}, {60, 180, 60}, {200, 200, 200}};
		const cv::Size size(80, 64);
		cv::Mat quadrant;
		const cv::Mat image = Quadrants(size, colours, 3.0, 99, &quadrant);
		const cv::Mat clean = Quadrants(size, colours, 0.0, 99);

		const MeanShift ms(8, 16);
		const MeanShift::Filtered filtered = ms.MSFiltering(BgrToLab(image));
		const MeanShift::Regions regions = ms.MSSegmentation(filtered.lab, 20);
		std::printf("%.2f steps per pixel, at most %d; %d regions grown, %d after merging\n", filtered.mean_steps,
		            filtered.max_steps, regions.grown, regions.Count());
		CHECK(filtered.unconverged == 0);
		CHECK(regions.Count() == 4);
		CheckRegions(regions, filtered.lab);

		// Each quadrant is exactly one region, with its colour
		int label[4];
		for (int q = 0; q < 4; ++q)
		{
			const cv::Point first((q % 2) * size.width / 2, (q / 2) * size.height / 2);
			label[q] = regions.labels.at<int>(first);
			CHECK(cv::countNonZero((regions.labels == label[q]) != (quadrant == q)) == 0);
			if (label[q] >= 0 && label[q] < regions.Count())
				CHECK(Distance(regions.colours[label[q]], LabOf(colours[q])) < 1.0);
		}
		CHECK(cv::norm(LabToBgr(regions.Image()), clean, cv::NORM_INF) <= 3.0);

		// Without noise the segmentation alone (no filtering) is exact
		const MeanShift::Regions direct = ms.MSSegmentation(BgrToLab(clean), 20);
		CHECK(direct.Count() == 4 && direct.grown == 4);
		CHECK(cv::norm(LabToBgr(direct.Image()), clean, cv::NORM_INF) <= 1.0);

		// The boundaries are drawn on the last row and column of each quadrant
		const cv::Mat drawn = DrawBoundaries(clean, direct.labels, cv::Vec3b(1, 2, 3));
		int misplaced = 0;
		for (int r = 0; r < size.height; ++r)
		{
			for (int c = 0; c < size.width; ++c)
			{
				const bool expected = r == size.height / 2 - 1 || c == size.width / 2 - 1;
				misplaced += (drawn.at<cv::Vec3b>(r, c) == cv::Vec3b(1, 2, 3)) != expected;
			}
		}
		CHECK(misplaced == 0);
	}

	// Distances are measured in L*a*b* of the BGR data (the 2019 program
	// applied RGB2Lab to BGR images, swapping red and blue)
	void TestColourDistance()
	{
		// Two blues 5.7 apart in L*a*b*; 31 apart if R and B are swapped
		const cv::Vec3b left(251, 28, 0), right(249, 23, 72);
		const double distance = Distance(LabOf(left), LabOf(right));
		std::printf("distance between the two colours: %.2f\n", distance);
		CHECK(distance < 8.0);

		cv::Mat image(40, 60, CV_8UC3, cv::Scalar(left[0], left[1], left[2]));
		image.colRange(30, 60).setTo(cv::Scalar(right[0], right[1], right[2]));
		const MeanShift ms(8, 16);
		CHECK(ms.MSSegmentation(ms.MSFiltering(BgrToLab(image)).lab, 20).Count() == 1);

		// In the swapped space the same image splits in two
		cv::Mat scaled, swapped;
		image.convertTo(scaled, CV_32F, 1.0 / 255.0);
		cv::cvtColor(scaled, swapped, cv::COLOR_RGB2Lab);
		CHECK(ms.MSSegmentation(ms.MSFiltering(swapped).lab, 20).Count() == 2);

		// A pure blue square on grey keeps its colour through both conversions
		cv::Mat patch(30, 30, CV_8UC3, cv::Scalar(128, 128, 128));
		patch(cv::Rect(10, 10, 10, 10)).setTo(cv::Scalar(255, 0, 0));
		const MeanShift::Regions regions = ms.MSSegmentation(ms.MSFiltering(BgrToLab(patch)).lab, 20);
		const cv::Mat segmented = LabToBgr(regions.Image());
		CHECK(regions.Count() == 2);
		CHECK(cv::norm(segmented, patch, cv::NORM_INF) <= 1.0);
	}

	// The filter for one pixel, written out separately from MeanShift.cpp:
	// the window is centred on the current mean (it moves), and the
	// iterations go on until both the position and the colour settle
	cv::Vec3f FilterPixel(const cv::Mat& lab, int row, int col, double hs, double hr, cv::Point2f& mode)
	{
		double x = col, y = row;
		cv::Vec3d colour(lab.at<cv::Vec3f>(row, col));
		for (int step = 0; step < MeanShift::kMaxSteps; ++step)
		{
			double sx = 0.0, sy = 0.0;
			cv::Vec3d sum(0.0, 0.0, 0.0);
			int n = 0;
			for (int r = 0; r < lab.rows; ++r)
			{
				for (int c = 0; c < lab.cols; ++c)
				{
					if ((c - x) * (c - x) + (r - y) * (r - y) > hs * hs)
						continue;
					const cv::Vec3d d = cv::Vec3d(lab.at<cv::Vec3f>(r, c)) - colour;
					if (d.dot(d) >= hr * hr)
						continue;
					sx += c;
					sy += r;
					sum += cv::Vec3d(lab.at<cv::Vec3f>(r, c));
					++n;
				}
			}
			if (n == 0)
				break;
			const double nx = sx / n, ny = sy / n;
			const cv::Vec3d next = sum / n;
			const cv::Vec3d shift = next - colour;
			const bool settled = (nx - x) * (nx - x) + (ny - y) * (ny - y) <= 0.09 && shift.dot(shift) <= 0.09;
			x = nx;
			y = ny;
			colour = next;
			if (settled)
				break;
		}
		mode = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
		return cv::Vec3f(colour);
	}

	void TestConvergence()
	{
		const cv::Mat lab = BgrToLab(Scene(cv::Size(56, 44), 5));
		const MeanShift ms(8, 16);
		const MeanShift::Filtered filtered = ms.MSFiltering(lab);
		std::printf("%.2f steps per pixel, at most %d, %d not converged\n", filtered.mean_steps, filtered.max_steps,
		            filtered.unconverged);
		CHECK(filtered.unconverged == 0);
		CHECK(filtered.mean_steps > 1.0);

		int compared = 0, matching = 0;
		for (int r = 0; r < lab.rows; r += 3)
		{
			for (int c = 0; c < lab.cols; c += 3)
			{
				cv::Point2f mode;
				const cv::Vec3f expected = FilterPixel(lab, r, c, 8, 16, mode);
				const cv::Vec2f got_mode = filtered.modes.at<cv::Vec2f>(r, c);
				++compared;
				matching += Distance(filtered.lab.at<cv::Vec3f>(r, c), expected) < 1e-3 &&
				            std::abs(got_mode[0] - mode.x) < 1e-3 && std::abs(got_mode[1] - mode.y) < 1e-3;
			}
		}
		std::printf("%d of %d sampled pixels match the separate implementation\n", matching, compared);
		// Rounding could tip the stopping test of a rare pixel either way
		CHECK(matching >= compared * 99 / 100);
	}

	// Flipping the image flips the filtered result: the window is
	// symmetric (the 2019 window was [j - 8, j + 8) and did not move)
	void TestSymmetry()
	{
		const cv::Mat lab = BgrToLab(Scene(cv::Size(48, 36), 17));
		const MeanShift ms(8, 16);
		const cv::Mat filtered = ms.MSFiltering(lab).lab;
		for (int code : {0, 1, -1, 2})
		{
			cv::Mat input, expected;
			if (code == 2)
			{
				cv::transpose(lab, input);
				cv::transpose(filtered, expected);
			}
			else
			{
				cv::flip(lab, input, code);
				cv::flip(filtered, expected, code);
			}
			const cv::Mat result = ms.MSFiltering(input).lab;
			cv::Mat difference;
			cv::absdiff(result, expected, difference);
			const cv::Mat largest = difference.reshape(1, static_cast<int>(difference.total()));
			cv::Mat per_pixel;
			cv::reduce(largest, per_pixel, 1, cv::REDUCE_MAX);
			const int differing = cv::countNonZero(per_pixel > 1e-3);
			double worst = 0.0;
			cv::minMaxLoc(per_pixel, nullptr, &worst);
			std::printf("%s: %d of %d pixels differ by more than 0.001 (at most %.4f)\n",
			            code == 2 ? "transpose" : code == 0 ? "vertical flip" : code == 1 ? "horizontal flip" : "both flips",
			            differing, static_cast<int>(per_pixel.total()), worst);
			CHECK(differing <= static_cast<int>(per_pixel.total()) / 100);
		}
	}

	// Regions under min_region pixels are merged into the neighbour with the
	// closest colour
	void TestSmallRegions()
	{
		const MeanShift ms(8, 16);

		// A grey background with a 10x10 red square and five 3x3 green specks
		cv::Mat image(60, 60, CV_8UC3, cv::Scalar(128, 128, 128));
		image(cv::Rect(40, 40, 10, 10)).setTo(cv::Scalar(0, 0, 255));
		const cv::Point specks[] = {{5, 5}, {20, 8}, {8, 30}, {30, 25}, {50, 10}};
		for (const cv::Point& p : specks)
			image(cv::Rect(p.x, p.y, 3, 3)).setTo(cv::Scalar(0, 255, 0));
		const cv::Mat lab = BgrToLab(image);

		CHECK(ms.MSSegmentation(lab, 0).Count() == 7);
		CHECK(ms.MSSegmentation(lab, 9).Count() == 7);
		const MeanShift::Regions merged = ms.MSSegmentation(lab, 20);
		CHECK(merged.grown == 7 && merged.Count() == 2);
		CheckRegions(merged, lab);
		const int background = merged.labels.at<int>(0, 0);
		for (const cv::Point& p : specks)
			CHECK(merged.labels.at<int>(p.y + 1, p.x + 1) == background);
		const int square = merged.labels.at<int>(45, 45);
		CHECK(square != background && merged.sizes[square] == 100);
		CHECK(ms.MSSegmentation(lab, 101).Count() == 1);

		// A speck on the border of two regions joins the one closer in colour
		cv::Mat halves(40, 40, CV_8UC3, cv::Scalar(50, 50, 50));
		halves.colRange(20, 40).setTo(cv::Scalar(230, 230, 230));
		halves(cv::Rect(19, 18, 3, 3)).setTo(cv::Scalar(100, 100, 100));
		const cv::Mat halves_lab = BgrToLab(halves);
		CHECK(ms.MSSegmentation(halves_lab, 0).Count() == 3);
		const MeanShift::Regions two = ms.MSSegmentation(halves_lab, 20);
		CHECK(two.Count() == 2);
		CHECK(two.labels.at<int>(19, 20) == two.labels.at<int>(0, 0));

		// A region with no neighbour (the whole image) stays, whatever its size
		const MeanShift::Regions single = ms.MSSegmentation(BgrToLab(cv::Mat(3, 3, CV_8UC3, cv::Scalar(9, 99, 199))), 20);
		CHECK(single.Count() == 1 && single.sizes[0] == 9);
	}

	void TestSampleImages()
	{
		const char* files[] = {
			"meanshift/S00-150x150.png", "meanshift/S02-150x150.png", "meanshift/S03-150x150.png",
			"meanshift/set1Seq2_L-150x150.png", "meanshift/set2Seq1_L-150x150.png",
		};
		const MeanShift ms(8, 16);
		for (const char* file : files)
		{
			const cv::Mat image = ReadSample(file);
			if (image.empty())
				continue;
			const MeanShift::Filtered filtered = ms.MSFiltering(BgrToLab(image));
			CHECK(filtered.unconverged == 0);
			for (int min_region : {20, 50})
			{
				const MeanShift::Regions regions = ms.MSSegmentation(filtered.lab, min_region);
				CheckRegions(regions, filtered.lab);
				CHECK(*std::min_element(regions.sizes.begin(), regions.sizes.end()) >= min_region);
				CHECK(regions.Count() >= 2 && regions.Count() < regions.grown);
				if (min_region == 20)
					std::printf("%-34s %.2f steps per pixel (at most %d), %d regions, %d after merging\n", file,
					            filtered.mean_steps, filtered.max_steps, regions.grown, regions.Count());
			}
		}
	}

	void TestArguments()
	{
		CHECK(Throws([] { MeanShift(0, 16); }));
		CHECK(Throws([] { MeanShift(8, -1); }));
		CHECK(Throws([] { MeanShift(std::nan(""), 16); }));
		CHECK(Throws([] { MeanShift(8, std::numeric_limits<double>::infinity()); }));

		const MeanShift ms(8, 16);
		CHECK(Throws([&] { ms.MSFiltering(cv::Mat()); }));
		CHECK(Throws([&] { ms.MSFiltering(cv::Mat(4, 4, CV_8UC3, cv::Scalar::all(0))); }));
		CHECK(Throws([&] { ms.MSSegmentation(cv::Mat(4, 4, CV_32FC3, cv::Scalar::all(0)), -1); }));
		CHECK(Throws([] { BgrToLab(cv::Mat(4, 4, CV_32FC3, cv::Scalar::all(0))); }));
		CHECK(Throws([] { BgrToLab(cv::Mat(4, 4, CV_8UC2, cv::Scalar::all(0))); }));
		CHECK(Throws([] { DrawBoundaries(cv::Mat(4, 4, CV_8UC3), cv::Mat(5, 5, CV_32S), cv::Vec3b()); }));

		// One pixel: its own mode after one step, and one region
		const MeanShift::Filtered one = ms.MSFiltering(BgrToLab(cv::Mat(1, 1, CV_8UC3, cv::Scalar(10, 20, 30))));
		CHECK(one.unconverged == 0 && one.max_steps == 1);
		CHECK(ms.MSSegmentation(one.lab, 20).Count() == 1);
	}

	const check::Test kTests[] = {
		{"lab_conversion", TestLabConversion},
		{"piecewise_constant", TestPiecewiseConstant},
		{"colour_distance", TestColourDistance},
		{"convergence", TestConvergence},
		{"symmetry", TestSymmetry},
		{"small_regions", TestSmallRegions},
		{"sample_images", TestSampleImages},
		{"arguments", TestArguments},
	};
}

int main(int argc, char** argv)
{
	return check::RunTests(kTests, argc, argv);
}
