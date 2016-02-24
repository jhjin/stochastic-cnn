package = "stcunn"
version = "scm-1"

source = {
   url = "git://github.com/jhjin/stochastic-cnn.git",
}

description = {
   summary = "Stochastic Feedforward Model for CUDA Neural Network",
   detailed = [[
   ]],
   homepage = "https://github.com/jhjin/stochastic-cnn",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "cutorch >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cd stcunn && cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN) install
]],
   install_command = "cd stcunn/build"
}
