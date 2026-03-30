base_eval_path <- "/Users/akatsurada/Documents/INSPER/StatisticsII/Aula7_lab/base_avaliacao.rds"
obj <- readRDS(base_eval_path)

class(obj)         # e.g., "list", "array", "matrix", "data.frame"
if (is.list(obj)) names(obj)
str(obj, max.level = 2)  # compact structure
if (!is.list(obj)) {
  dim(obj); length(obj); head(as.data.frame(obj), 3)
}