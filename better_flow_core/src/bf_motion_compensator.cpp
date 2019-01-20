#include <better_flow/common.h>
#include <better_flow/dvs_flow.h>
#include <better_flow/opencl_driver.h>


#define EVENT_WIDTH 50000
#define TIME_WIDTH 0.2

float time_refresh = 0.033;
unsigned long long int event_refresh = 20000;

bool manual = false;
bool quiet = false;
char *file = NULL;
char *outFileName = NULL;
bool gpu = false;
bool img = false;
bool video = false;
bool stm_disable = false;
bool bufferize_file = false;
std::string img_prefix = "./";
std::string video_name = "./out.avi";
int video_fps = 60;

static void
lPrintVersion() {
    printf("DVS flow estimator (better flow), %s (build %s @ %s)\n",
           BF_VERSION, __DATE__, __TIME__
           );
    printf("\tCompiled with maximum event memory of %i events\n\tand slice size of %f seconds.\n",
           EVENT_WIDTH, TIME_WIDTH
           );
}


static void
usage(int ret) {
    lPrintVersion();
    printf("\nusage: bf_motion_compensator\n");
    printf("    [--refresh-time={0.0 - inf}]\t\tRun processing when at least this amount of time (floatimg point,\n");
    printf("                                \t\tseconds) has passed since the last processing, (default = %f)\n", time_refresh);
    printf("    [--refresh-event-count={0 - inf}]\t\tRun processing when at least this number of new events has\n");
    printf("                                     \t\tarrived since the last processing (default = %llu)\n", event_refresh);
    printf("    [-i/--interactive]\tEnable interactive mode\n");
    printf("    [-G]\t\t\t\tUse GPU support\n");
    printf("    [--stm-disable]\t\t\t\tDo not use previous estimate as a starting point for a new estimate\n");
    printf("    [--img]\t\t\t\tOutput flow images after every iteration\n");
    printf("    [--img-prefix <name>]\t\t\t\tSpecify prefix for the generated image files (default = %s)\n", img_prefix.c_str());
    printf("    [--video]\t\t\t\tOutput a video with flow frames\n");
    printf("    [--video-name <name>]\t\t\t\tSpecify the name of the video file (default = %s)\n", video_name.c_str());
    printf("    [--video-fps=<value>]\t\t\t\tSpecify video framerate (default = %i)\n", video_fps);
    printf("    [--bufferize-file]\t\t\t\tRead input file to the buffer first (useful for performance testing)\n");
    printf("    [--quiet]\t\t\t\tSuppress all output\n");
    printf("    [-o <name>/--outfile=<name>]\tOutput filename (may be \"-\" for standard output)\n");
    printf("    [--version]\t\t\t\tPrint better flow version\n");
    printf("    <file to process or \"-\" for stdin>\n");
    exit(ret);
}


int main (int argc, char *argv[]) {
    // CLI parameters
    if (argc == 1) usage(1);
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help"))
            usage(0);
        else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--version")) {
            lPrintVersion();
            return 0;
        }
        else if (!strcmp(argv[i], "--quiet"))
            quiet = true;
        else if (!strncmp(argv[i], "--refresh-time=", 15))
            time_refresh = atof(argv[i] + 15);
        else if (!strncmp(argv[i], "--refresh-event-count=", 22))
            event_refresh = atoi(argv[i] + 22);        
        else if (!strcmp(argv[i], "-G"))
            gpu = true;
        else if (!strcmp(argv[i], "-i"))
            manual = true;
        else if (!strcmp(argv[i], "--interactive"))
            manual = true;       
        else if (!strcmp(argv[i], "--bufferize-file"))
            bufferize_file = true;
        else if (!strcmp(argv[i], "--stm-disable"))
            stm_disable = true;
        else if (!strcmp(argv[i], "--img"))
            img = true;
        else if (!strcmp(argv[i], "--img-prefix")) {
            if (++i == argc) {
                fprintf(stderr, "No output file specified after --img-prefix option.\n");
                usage(1);
            }
            img_prefix = argv[i];
        }
        else if (!strcmp(argv[i], "--video"))
            video = true;
        else if (!strcmp(argv[i], "--video-name")) {
            if (++i == argc) {
                fprintf(stderr, "No output file specified after --video-name option.\n");
                usage(1);
            }
            video_name = argv[i];
        }
        else if (!strncmp(argv[i], "--video-fps=", 12))
            video_fps = atoi(argv[i] + 12);
        else if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                fprintf(stderr, "No output file specified after -o option.\n");
                usage(1);
            }
            outFileName = argv[i];
        }
        else if (!strncmp(argv[i], "--outfile=", 10))
            outFileName = argv[i] + strlen("--outfile=");
        else if (!strcmp(argv[i], "-")) {}
		else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option \"%s\".\n", argv[i]);
            usage(1);
        }
        else {
            if (file != NULL) {
                fprintf(stderr, "Multiple input files specified on command "
                        "line: \"%s\" and \"%s\".\n", file, argv[i]);
                usage(1);
            }
            else
                file = argv[i];
        }
    }

    if (gpu)
        OpenCLDriver::init();

    DVS_flow<EVENT_WIDTH, FROM_SEC(TIME_WIDTH)> estimator(event_refresh, FROM_SEC(time_refresh));
    
    if (outFileName != NULL)
        estimator.set_accumulate(); // This will enable event bufferization
    
    if (manual)
        estimator.set_manual_mode(true);

    if (img)
        estimator.set_generate_pictures(true, img_prefix);

    if (video)
        estimator.set_generate_video(true, video_name, video_fps);

    if (stm_disable)
        estimator.set_stm_disable(true);


    // Read file to the buffer first
    if (bufferize_file) {
        LinearEventCloud ec;
        EventFile::from_file(&ec, file);

        clock_t begin = std::clock();
        clock_t begin_slice = std::clock();

        ull i = 0;
        for (auto &e : ec) {
            ++i;
            bool processed = estimator.add_event(e);
            if (processed) {
                clock_t end_slice = std::clock();
                std::cout << float(i * 100) / float(ec.size()) << " %\t"
                          << i << "\t"
                          << (double(end_slice - begin_slice) / CLOCKS_PER_SEC) << " sec\t"
                          << estimator.get_buf_size() << " events\t"
                          << double(estimator.get_time_diff()) / 1000000000.0 << " slice_td\t"
                          << double(estimator.get_buf_time_diff()) / 1000000000.0 << " buffer_td\n";
                begin_slice = std::clock();
            }
        }

        clock_t end = std::clock();
        std::cout << "Toatal flow elapsed: " << double(end - begin) / CLOCKS_PER_SEC << " sec." << std::endl << std::flush;
    } else {
        std::cout << "Reading from file... (" << file << ")" << std::endl << std::flush;
        std::ifstream event_file(file, std::ifstream::in);

        ull i = 0;
        double t = 0;
        uint x = 0, y = 0;
        bool p = false;

        double t_0 = 0; // the earliest timestamp in the file
    
        if (event_file >> t_0 >> x >> y >> p) {
            ++i;
            Event e(y, x, FROM_SEC(0));
            estimator.add_event(e);
        }

        while (event_file >> t >> x >> y >> p) {
            t -= t_0;

            ++i;
            Event e(y, x, FROM_SEC(t));
            estimator.add_event(e);
        }

        event_file.close();
        std::cout << "Read and processed " << i << " events" << std::endl << std::flush;
    }

    estimator.recompute(); // Ensure that *every* event has been processed

    if (outFileName != NULL) {
        LinearEventCloudTemplate<Event> accumulated = estimator.get_accumulated();
        EventFile::to_file_uv(&accumulated, outFileName);
    }

    return 0;
}
