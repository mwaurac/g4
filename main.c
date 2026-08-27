// plan: I want to parse the cli args like ./g4 --model <path>
// I would also like to parse other hparams like top-k, top-p, context length
// etc
//

#include <stdio.h>
#include <string.h>

typedef struct {
  const char *model_dir;
} cli_config;

static void usage() {
  fprintf(stderr, "Gemma4 E2B Inference\n");
  fprintf(stderr, "Options:\n\n");
  fprintf(stderr, "-m, --model <path>     Path to the model directory\n");
  fprintf(stderr, "-h, --help             Show this help message\n");
}

static int init_cfg(int argc, char *argv[], cli_config *cfg) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: -m requires an argument\n");
        return 0;
      }
      cfg->model_dir = argv[i];
    }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      return 0;
    } else {
      fprintf(stderr, "Error: Unknown option\n");
    }
  }
  return 1;
}
int main(int argc, char *argv[]) {
  cli_config cfg;

  if (!init_cfg(argc, argv, &cfg)) {
    usage();
    return 1;
  }
  return 0;
}
