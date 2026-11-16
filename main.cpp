// main.cpp
//
// Command-line program for three classical segmentation methods: Otsu
// thresholding with 2 classes, Otsu thresholding with 4 classes (Otsu.h),
// and mean shift filtering and segmentation (MeanShift.h). In the window,
// keys 1, 2 and 3 run them; with --no-gui the program runs the chosen
// methods once, prints their results and writes the result images.
//
// Michael Perez, CAP 4410 Computer Vision, Assignment 3, Florida Polytechnic
// University (fall 2019). Command-line options and the headless mode added
// in 2026.

#include "MeanShift.h"
#include "Otsu.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{
	enum class Method { All, Otsu2, Otsu4, MeanShift };

	struct Options
	{
		std::string image = "images/otsu4/blocks_L-150x150.png";
		std::string output_dir;  // where to write the result images; empty = do not write
		Method method = Method::All;
		bool method_given = false;
		double hs = 8.0;         // mean shift spatial bandwidth, pixels
		double hr = 16.0;        // mean shift colour bandwidth, L*a*b* units
		int min_region = 20;     // mean shift regions smaller than this are merged
		bool gui = true;
	};

	constexpr double kMaxSpatialBandwidth = 50.0;
	constexpr double kMaxColourBandwidth = 100.0;
	constexpr int kMaxMinRegion = 1000000;
	const cv::Vec3b kBoundaryColour(0, 0, 255);  // red, in BGR order

	// One result image, with its window title and the end of its file name
	struct Output
	{
		std::string window;
		std::string suffix;
		cv::Mat image;
	};

	using Clock = std::chrono::steady_clock;

	void PrintUsage(std::ostream& out, const char* program)
	{
		out << "Usage: " << program << " [options] [image]\n"
		    << "\n"
		    << "Segments an image with Otsu thresholding (2 or 4 classes) and mean shift\n"
		    << "(default image: images/otsu4/blocks_L-150x150.png).\n"
		    << "\n"
		    << "Options:\n"
		    << "  --method M       otsu2, otsu4, meanshift or all (default all)\n"
		    << "  --hs N           mean shift spatial bandwidth in pixels, 1 to " << kMaxSpatialBandwidth
		    << " (default 8)\n"
		    << "  --hr N           mean shift colour bandwidth in L*a*b* units, 1 to " << kMaxColourBandwidth
		    << " (default 16)\n"
		    << "  --min-region N   merge mean shift regions smaller than N pixels, 0 to " << kMaxMinRegion
		    << "\n"
		    << "                   (default 20; 0 merges nothing)\n"
		    << "  --out DIR        write the result images to DIR\n"
		    << "  --no-gui         do not open windows: run the method(s), write the result\n"
		    << "                   images (to ./output unless --out is given) and exit\n"
		    << "  -h, --help       show this message\n"
		    << "\n"
		    << "In the window, press 1 for Otsu with 2 classes, 2 for Otsu with 4 classes,\n"
		    << "3 for mean shift, and q or Esc to quit. A method given with --method also\n"
		    << "runs at the start.\n";
	}

	// Parses an integer from `low` to `high`
	bool ParseInteger(const std::string& text, int low, int high, int& value)
	{
		try
		{
			std::size_t used = 0;
			const long long parsed = std::stoll(text, &used);
			if (used != text.size() || parsed < low || parsed > high)
				return false;
			value = static_cast<int>(parsed);
			return true;
		}
		catch (const std::exception&)
		{
			return false;
		}
	}

	// Parses a number from `low` to `high`
	bool ParseNumber(const std::string& text, double low, double high, double& value)
	{
		try
		{
			std::size_t used = 0;
			const double parsed = std::stod(text, &used);
			if (used != text.size() || !std::isfinite(parsed) || parsed < low || parsed > high)
				return false;
			value = parsed;
			return true;
		}
		catch (const std::exception&)
		{
			return false;
		}
	}

	bool ParseMethod(const std::string& text, Method& method)
	{
		if (text == "all")
			method = Method::All;
		else if (text == "otsu2")
			method = Method::Otsu2;
		else if (text == "otsu4")
			method = Method::Otsu4;
		else if (text == "meanshift")
			method = Method::MeanShift;
		else
			return false;
		return true;
	}

	enum class ParseResult { Run, Help, Error };

	ParseResult ParseArguments(int argc, char** argv, Options& options)
	{
		bool image_given = false;
		for (int i = 1; i < argc; ++i)
		{
			const std::string arg = argv[i];
			const bool has_value = i + 1 < argc;
			if (arg == "-h" || arg == "--help")
				return ParseResult::Help;

			if (arg == "--no-gui")
			{
				options.gui = false;
			}
			else if (arg == "--method")
			{
				if (!has_value || !ParseMethod(argv[i + 1], options.method))
				{
					std::cerr << "--method must be otsu2, otsu4, meanshift or all\n";
					return ParseResult::Error;
				}
				options.method_given = true;
				++i;
			}
			else if (arg == "--hs" || arg == "--hr")
			{
				const bool spatial = arg == "--hs";
				double& target = spatial ? options.hs : options.hr;
				const double limit = spatial ? kMaxSpatialBandwidth : kMaxColourBandwidth;
				if (!has_value || !ParseNumber(argv[i + 1], 1.0, limit, target))
				{
					std::cerr << arg << " needs a number from 1 to " << limit << "\n";
					return ParseResult::Error;
				}
				++i;
			}
			else if (arg == "--min-region")
			{
				if (!has_value || !ParseInteger(argv[i + 1], 0, kMaxMinRegion, options.min_region))
				{
					std::cerr << "--min-region needs an integer from 0 to " << kMaxMinRegion << "\n";
					return ParseResult::Error;
				}
				++i;
			}
			else if (arg == "--out")
			{
				if (!has_value)
				{
					std::cerr << "--out needs a directory\n";
					return ParseResult::Error;
				}
				options.output_dir = argv[++i];
			}
			else if (arg.size() > 1 && arg[0] == '-')
			{
				std::cerr << "Unknown option: " << arg << "\n";
				return ParseResult::Error;
			}
			else if (image_given)
			{
				std::cerr << "Only one image can be given\n";
				return ParseResult::Error;
			}
			else
			{
				options.image = arg;
				image_given = true;
			}
		}
		return ParseResult::Run;
	}

	std::string Fixed(double value, int precision)
	{
		std::ostringstream text;
		text << std::fixed << std::setprecision(precision) << value;
		return text.str();
	}

	double SecondsSince(Clock::time_point start)
	{
		return std::chrono::duration<double>(Clock::now() - start).count();
	}

	std::string Join(const std::vector<int>& values)
	{
		std::string text;
		for (std::size_t i = 0; i < values.size(); ++i)
			text += (i ? ", " : "") + std::to_string(values[i]);
		return text;
	}

	// The number of pixels in each class, as "n0, n1, ..."
	std::string ClassSizes(const otsu::Result& result)
	{
		std::vector<int> sizes;
		for (double size : result.class_sizes)
			sizes.push_back(static_cast<int>(size));
		return Join(sizes);
	}

	// Key 1: two classes on the grayscale image, and for a colour image also
	// two classes on each channel (what the 2019 program showed)
	std::vector<Output> RunOtsu2(const cv::Mat& image)
	{
		Clock::time_point start = Clock::now();
		const cv::Mat gray = otsu::ToGray(image);
		const otsu::Result result = otsu::FindThresholds(gray, 2);
		std::vector<Output> outputs = {{"Otsu, 2 classes", "otsu2", otsu::ClassImage(gray, result.thresholds)}};
		std::cout << "Otsu, 2 classes: threshold " << result.thresholds[0] << ", eta " << Fixed(result.Eta(), 3)
		          << ", class sizes " << ClassSizes(result) << " (" << Fixed(SecondsSince(start), 3) << " s)\n";

		if (!otsu::IsGray(image))
		{
			start = Clock::now();
			std::vector<otsu::Result> channels;
			outputs.push_back({"Otsu, 2 classes per channel", "otsu2_channels", otsu::ThresholdChannels(image, &channels)});
			std::cout << "Otsu, 2 classes per channel: thresholds B " << channels[0].thresholds[0] << ", G "
			          << channels[1].thresholds[0] << ", R " << channels[2].thresholds[0] << ", eta "
			          << Fixed(channels[0].Eta(), 3) << ", " << Fixed(channels[1].Eta(), 3) << ", "
			          << Fixed(channels[2].Eta(), 3) << " (" << Fixed(SecondsSince(start), 3) << " s)\n";
		}
		return outputs;
	}

	// Key 2: four classes on the grayscale image
	std::vector<Output> RunOtsu4(const cv::Mat& image)
	{
		const Clock::time_point start = Clock::now();
		const cv::Mat gray = otsu::ToGray(image);
		const otsu::Result result = otsu::FindThresholds(gray, 4);
		std::cout << "Otsu, 4 classes: thresholds " << Join(result.thresholds) << ", eta " << Fixed(result.Eta(), 3)
		          << ", class sizes " << ClassSizes(result) << " (" << Fixed(SecondsSince(start), 3) << " s)\n";
		return {{"Otsu, 4 classes", "otsu4", otsu::ClassImage(gray, result.thresholds)}};
	}

	// Key 3: mean shift filtering, then segmentation of the filtered image.
	// The time printed covers these two steps. The clock starts after the
	// conversion to L*a*b*, because the first conversion in a process also
	// builds OpenCV's lookup tables for it, which takes a few tenths of a
	// second.
	std::vector<Output> RunMeanShift(const cv::Mat& image, const Options& options)
	{
		const cv::Mat lab = BgrToLab(image);
		const Clock::time_point start = Clock::now();
		const MeanShift MSProc(options.hs, options.hr);
		const MeanShift::Filtered filtered = MSProc.MSFiltering(lab);
		const MeanShift::Regions regions = MSProc.MSSegmentation(filtered.lab, options.min_region);
		const double seconds = SecondsSince(start);

		std::cout << "Mean shift (hs " << options.hs << ", hr " << options.hr << "): "
		          << Fixed(filtered.mean_steps, 2) << " steps per pixel on average, at most " << filtered.max_steps;
		if (filtered.unconverged > 0)
			std::cout << " (" << filtered.unconverged << (filtered.unconverged == 1 ? " pixel" : " pixels")
			          << " did not converge)";
		std::cout << "; " << regions.grown << (regions.grown == 1 ? " region" : " regions");
		if (options.min_region > 1)
			std::cout << ", " << regions.Count() << " after merging regions under " << options.min_region
			          << " pixels";
		std::cout << " (" << Fixed(seconds, 2) << " s)\n";

		return {{"Mean shift filtered", "meanshift_filtered", LabToBgr(filtered.lab)},
		        {"Mean shift segmented", "meanshift_segmented", LabToBgr(regions.Image())},
		        {"Mean shift boundaries", "meanshift_boundaries", DrawBoundaries(image, regions.labels, kBoundaryColour)}};
	}

	std::vector<Output> Run(Method method, const cv::Mat& image, const Options& options)
	{
		std::vector<Output> outputs;
		auto append = [&outputs](std::vector<Output> more) {
			for (Output& output : more)
				outputs.push_back(std::move(output));
		};
		if (method == Method::All || method == Method::Otsu2)
			append(RunOtsu2(image));
		if (method == Method::All || method == Method::Otsu4)
			append(RunOtsu4(image));
		if (method == Method::All || method == Method::MeanShift)
			append(RunMeanShift(image, options));
		return outputs;
	}

	bool CreateOutputDirectory(const std::string& dir)
	{
		std::error_code error;
		std::filesystem::create_directories(dir, error);
		if (error)
		{
			std::cerr << "Could not create " << dir << ": " << error.message() << "\n";
			return false;
		}
		return true;
	}

	// Writes <dir>/<stem>_<suffix>.png for each output
	bool WriteOutputs(const std::vector<Output>& outputs, const std::string& dir, const std::string& stem)
	{
		const std::vector<int> png_options = {cv::IMWRITE_PNG_COMPRESSION, 9};
		bool ok = true;
		for (const Output& output : outputs)
		{
			const std::string path = (std::filesystem::path(dir) / (stem + "_" + output.suffix + ".png")).string();
			if (output.image.empty() || !cv::imwrite(path, output.image, png_options))
			{
				std::cerr << "Could not write " << path << "\n";
				ok = false;
			}
			else
			{
				std::cout << "Wrote " << path << "\n";
			}
		}
		return ok;
	}

	int RunGui(const cv::Mat& image, const Options& options, const std::string& stem)
	{
		const std::string window = "Image";
		cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
		cv::imshow(window, image);
		cv::waitKey(1);

		// Not every backend reports whether a window is still open, so only
		// watch for a closed window when the backend reports it as visible now.
		const bool can_detect_close = cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) >= 1;

		auto run = [&](Method method) {
			const std::vector<Output> outputs = Run(method, image, options);
			for (const Output& output : outputs)
				cv::imshow(output.window, output.image);
			if (!options.output_dir.empty())
				WriteOutputs(outputs, options.output_dir, stem);
		};
		if (options.method_given)
			run(options.method);

		std::cout << "Press 1 for Otsu with 2 classes, 2 for Otsu with 4 classes, 3 for mean shift,\n"
		          << "q or Esc to quit." << std::endl;
		while (true)
		{
			const int key = cv::waitKey(50);
			if (key == 'q' || key == 'Q' || key == 27)
				break;
			if (can_detect_close && cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1)
				break;

			if (key == '1')
				run(Method::Otsu2);
			else if (key == '2')
				run(Method::Otsu4);
			else if (key == '3')
				run(Method::MeanShift);
		}
		cv::destroyAllWindows();
		return 0;
	}
}

int main(int argc, char** argv)
{
	Options options;
	switch (ParseArguments(argc, argv, options))
	{
	case ParseResult::Help:
		PrintUsage(std::cout, argv[0]);
		return 0;
	case ParseResult::Error:
		PrintUsage(std::cerr, argv[0]);
		return 2;
	case ParseResult::Run:
		break;
	}
	if (!options.gui && options.output_dir.empty())
		options.output_dir = "output";

	const cv::Mat image = cv::imread(options.image, cv::IMREAD_COLOR);
	if (image.empty())
	{
		std::cerr << "Could not read the image '" << options.image << "'.\n";
		return 1;
	}
	std::cout << "Image: " << options.image << " (" << image.cols << "x" << image.rows << ")\n";

	// Create the output directory before any method runs, so that a bad
	// path is reported at once
	if (!options.output_dir.empty() && !CreateOutputDirectory(options.output_dir))
		return 1;
	const std::string stem = std::filesystem::path(options.image).stem().string();

	try
	{
		if (!options.gui)
			return WriteOutputs(Run(options.method, image, options), options.output_dir, stem) ? 0 : 1;
		return RunGui(image, options, stem);
	}
	catch (const cv::Exception& e)
	{
		std::cerr << "OpenCV error: " << e.what() << "\n";
		if (options.gui)
			std::cerr << "To run without windows, use --no-gui.\n";
		return 1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}
