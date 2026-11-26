# bihaba-serve Makefile — Created by Boy Barley (Bihaba — Team Pasai Aceh)
CXX        := clang++
CXXFLAGS   := -O3 -std=c++17 -fobjc-arc
FRAMEWORKS := -framework Metal -framework Foundation

BUILD_DIR  := build

.PHONY: all qwen35 transformer clean install uninstall

all: qwen35 transformer

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

qwen35: $(BUILD_DIR)
	@echo "▸ Compiling bihaba-qwen35 (Qwen3.5 Dense & 256-Expert MoE)..."
	$(CXX) $(CXXFLAGS) engines/qwen35/main.mm $(FRAMEWORKS) -o $(BUILD_DIR)/bihaba-qwen35
	@cp engines/qwen35/kernels.metal $(BUILD_DIR)/qwen35_kernels.metal
	@echo "✓ Built $(BUILD_DIR)/bihaba-qwen35"

transformer: $(BUILD_DIR)
	@echo "▸ Compiling bihaba-transformer (Qwen2.5 / Qwen3 / Llama / MNN-Pocket)..."
	$(CXX) $(CXXFLAGS) engines/transformer/main.mm $(FRAMEWORKS) -o $(BUILD_DIR)/bihaba-transformer
	@cp engines/transformer/kernels.metal $(BUILD_DIR)/transformer_kernels.metal
	@echo "✓ Built $(BUILD_DIR)/bihaba-transformer"

install:
	@./install.sh

uninstall:
	@./uninstall.sh

clean:
	@rm -rf $(BUILD_DIR) python/bihaba_serve/__pycache__
	@echo "✓ Cleaned build artifacts"
