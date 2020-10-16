#include <test/cpp/tensorexpr/test_base.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace torch {
namespace jit {

using namespace torch::indexing;
using namespace torch::jit::tensorexpr;

void testKernel_1() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[3, 1], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

void testKernel_2() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[1, 5], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b =
      at::rand({3, 5}, TensorOptions(kCPU).dtype(at::kFloat)).transpose(0, 1);
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

void testKernel_3() {
  KernelScope kernel_scope;

  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[12, 2], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({10, 6}, TensorOptions(kCPU).dtype(at::kFloat))
               .index({Slice(None, None, 2), Slice(None, None, 2)});
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  Stmt* s = k.getCodeGenStmt();

  std::ostringstream oss;
  oss << *s;

  // Check the IR we produced
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

void testKernel_4() {
  // Test TensorExpr shape inference capabilities: it should only require shapes
  // for the inputs
  {
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[12, 2], device=cpu)):
        %2 : Tensor = aten::mul(%0, %1)
        %3 : Tensor = aten::mul(%0, %2)
        return (%3))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({10, 6}, TensorOptions(kCPU).dtype(at::kFloat))
                 .index({Slice(None, None, 2), Slice(None, None, 2)});
    auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = a * (a * b);
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    for (size_t i = 0; i < 5 * 3; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%0 : Float(8, 8, strides=[8, 1], device=cpu),
            %1 : Float(8, 8, strides=[8, 1], device=cpu)):
        %2 : Tensor = aten::mul(%0, %1)
        %3 : Tensor, %4 : Tensor = prim::ConstantChunk[dim=1,chunks=2](%2)
        %r : Tensor = aten::mul(%3, %4)
        return (%r))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({8, 4}, TensorOptions(kCPU).dtype(at::kFloat));
    auto t = torch::chunk(a * b, 2, 1);
    auto ref = t[0] * t[1];
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    CHECK_EQ(o.sizes()[0], 8);
    CHECK_EQ(o.sizes()[1], 4);
    for (size_t i = 0; i < 8 * 4; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // Test that shape inference handles aten::unsqueeze
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%a : Float(4, 2, strides=[2, 1], device=cpu),
            %b : Float(4, 3, 2, strides=[6, 2, 1], device=cpu),
            %c : Float(3, 2, 2, strides=[4, 2, 1], device=cpu)):
        %one : int = prim::Constant[value=1]()
        %minus_one : int = prim::Constant[value=-1]()
        %three : int = prim::Constant[value=3]()
        %minus_four : int = prim::Constant[value=-4]()
        %a1 : Tensor = aten::unsqueeze(%a, %one)        # new size: [4,1,2]
        %a2 : Tensor = aten::unsqueeze(%a1, %minus_one) # new size: [4,1,2,1]
        %b1 : Tensor = aten::unsqueeze(%b, %three)      # new size: [4,3,2,1]
        %c1 : Tensor = aten::unsqueeze(%c, %minus_four) # new size: [1,3,2,2]
        %ab : Tensor = aten::mul(%a2, %b1)         # expected size: [4,3,2,1]
        %abc : Tensor = aten::mul(%ab, %c1)        # expected size: [4,3,2,2]
        return (%abc))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({4, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({4, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto c = at::rand({3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({4, 3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = at::unsqueeze(at::unsqueeze(a, 1), -1) * at::unsqueeze(b, 3) *
        at::unsqueeze(c, -4);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b, c};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_mul)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();

    // Check sizes
    CHECK_EQ(o.sizes().size(), ref.sizes().size());
    size_t num_el = 1;
    for (size_t idx = 0; idx < ref.sizes().size(); idx++) {
      CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (size_t i = 0; i < num_el; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // Test that shape inference handles aten::cat
    KernelScope kernel_scope;

    const auto graph_string = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 7, 2, strides=[14, 2, 1], device=cpu),
            %c : Float(5, 9, 2, strides=[18, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Tensor = aten::cat(%inputs, %dim)               # new size: [5,19,2]
        return (%r))IR";
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto a = at::rand({5, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({5, 7, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto c = at::rand({5, 9, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({5, 19, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto ref = at::cat({a, b, c}, 1);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a, b, c};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_cat)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();

    // Check sizes
    CHECK_EQ(o.sizes().size(), ref.sizes().size());
    size_t num_el = 1;
    for (size_t idx = 0; idx < ref.sizes().size(); idx++) {
      CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      num_el *= ref.sizes()[idx];
    }

    // Check the contents
    for (size_t i = 0; i < num_el; i++) {
      CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
}

namespace {

std::string dtypeConstant(ScalarType scalar_type) {
  if (scalar_type == ScalarType::None) {
    return "None = prim::Constant()";
  } else {
    TemplateEnv env_dtype;
    env_dtype.d("scalar_type", static_cast<int>(scalar_type));
    return format("int = prim::Constant[value=${scalar_type}]()", env_dtype);
  }
}

at::Tensor iotaTensor(IntArrayRef sizes, const at::TensorOptions& options) {
  int64_t numel = std::accumulate(
      sizes.begin(), sizes.end(), 1, std::multiplies<int64_t>());
  std::vector<float> values(numel);
  std::iota(values.begin(), values.end(), 0);
  auto a = at::tensor(values, options);
  return a.reshape(sizes);
}

} // namespace

void testKernelSumAllAxes() {
  // Test lowering of sum on all axes.
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : ${dtype}
        %2 : Tensor = aten::sum(%0, %1)
        return (%2))IR";
  auto a = iotaTensor({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  for (auto scalar_type : {ScalarType::None, ScalarType::Double}) {
    KernelScope kernel_scope;
    TemplateEnv env;
    env.s("dtype", dtypeConstant(scalar_type));
    const auto graph_string = format(graph_template, env);

    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    auto o = at::empty({}, TensorOptions(kCPU));
    c10::optional<c10::ScalarType> dtype;
    if (scalar_type != ScalarType::None) {
      dtype = static_cast<c10::ScalarType>(scalar_type);
    }
    auto ref = a.sum(/*dtype=*/dtype);
    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    // Check the IR we produced
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    o = stack[0].toTensor();
    ASSERT_EQ(o.sizes(), ref.sizes());
    ASSERT_EQ(o.dtype(), ref.dtype());
    ASSERT_TRUE(at::allclose(o, ref));
  }
}

void testKernelSumOneAxis() {
  // Test lowering of sum on one axis.
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : int[] = prim::Constant[value=[${dim}]]()
        %2 : bool = prim::Constant[value=${keepdim}]()
        %3 : ${dtype}
        %4 : Tensor = aten::sum(%0, %1, %2, %3)
        return (%4))IR";
  auto a = iotaTensor({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  for (int dim = -a.dim(); dim < a.dim(); ++dim) {
    for (bool keepdim : {false, true}) {
      for (auto scalar_type : {ScalarType::None, ScalarType::Double}) {
        KernelScope kernel_scope;
        TemplateEnv env;
        env.d("dim", dim);
        env.d("keepdim", keepdim);
        env.s("dtype", dtypeConstant(scalar_type));
        const auto graph_string = format(graph_template, env);

        auto graph = std::make_shared<Graph>();
        parseIR(graph_string, &*graph);

        auto o = at::empty({}, TensorOptions(kCPU));
        c10::optional<c10::ScalarType> dtype;
        if (scalar_type != ScalarType::None) {
          dtype = static_cast<c10::ScalarType>(scalar_type);
        }
        auto ref = a.sum({dim}, /*keepdim=*/keepdim, /*dtype=*/dtype);
        TensorExprKernel k(graph);
        std::vector<at::Tensor> inputs = {a};
        Stmt* s = k.getCodeGenStmt();

        std::ostringstream oss;
        oss << *s;

        // Check the IR we produced
        const std::string& verification_pattern =
            R"IR(
# CHECK: int v = 0
# CHECK: int v_1 = 0
# CHECK: input1)IR";
        torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

        std::vector<IValue> stack = fmap<IValue>(inputs);
        k.run(stack);
        o = stack[0].toTensor();
        ASSERT_EQ(o.sizes(), ref.sizes());
        ASSERT_EQ(o.dtype(), ref.dtype());
        ASSERT_TRUE(at::allclose(o, ref));
      }
    }
  }
}

void testKernelSumMultipleAxes() {
  // Test lowering of sum on multiple axes.
  const auto graph_template = R"IR(
      graph(%0 : Float(2, 3, 2, 3, strides=[18, 6, 3, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim1}]()
        %2 : int = prim::Constant[value=${dim2}]()
        %3 : int[] = prim::ListConstruct(%1, %2)
        %4 : bool = prim::Constant[value=${keepdim}]()
        %5 : ${dtype}
        %6 : Tensor = aten::sum(%0, %3, %4, %5)
        return (%6))IR";
  auto a = iotaTensor({2, 3, 2, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  // Only iterate over positive values of axes to keep the running time
  // reasonable, since the number of pairs is quadratic.
  for (int dim1 = 0; dim1 < a.dim(); ++dim1) {
    for (int dim2 = dim1 + 1; dim2 < a.dim(); ++dim2) {
      for (bool keepdim : {false, true}) {
        KernelScope kernel_scope;
        TemplateEnv env;
        env.d("dim1", dim1);
        env.d("dim2", dim2);
        env.d("keepdim", keepdim);
        env.s("dtype", dtypeConstant(ScalarType::None));
        const auto graph_string = format(graph_template, env);

        auto graph = std::make_shared<Graph>();
        parseIR(graph_string, &*graph);

        auto o = at::empty({}, TensorOptions(kCPU));
        auto ref = a.sum(IntArrayRef{dim1, dim2}, /*keepdim=*/keepdim);
        TensorExprKernel k(graph);
        std::vector<at::Tensor> inputs = {a};
        Stmt* s = k.getCodeGenStmt();

        std::ostringstream oss;
        oss << *s;

        // Check the IR we produced
        const std::string& verification_pattern =
            R"IR(
# CHECK: int v = 0
# CHECK: int v_1 = 0
# CHECK: int v_2 = 0
# CHECK: int v_3 = 0
# CHECK: input1)IR";
        torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

        std::vector<IValue> stack = fmap<IValue>(inputs);
        k.run(stack);
        o = stack[0].toTensor();
        ASSERT_EQ(o.sizes(), ref.sizes());
        ASSERT_EQ(o.dtype(), ref.dtype());
        ASSERT_TRUE(at::allclose(o, ref));
      }
    }
  }
}

// This test and the following ones testing Softmax only tests with dim set
// to one of the valid input dimensions. It does not test with dim=None
// because that is supposed to be deprecated.
void testKernelSoftmax2D() {
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %2 : int = prim::Constant[value=7]()
        %3 : Tensor = aten::softmax(%0, %1, %2)
        return (%3))IR";

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  const std::string& verification_template =
      R"IR(
        # CHECK: for (int i0 = 0; i0 < 5
        # CHECK-NEXT: for (int i1 = 0; i1 < 3
        # CHECK-NEXT: input1
        # CHECK: for (int i${other_dim}_1 = 0; i${other_dim}_1 < ${other_dim_size}
        # CHECK: for (int i${softmax_dim}_1 = 0; i${softmax_dim}_1 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_max
        # CHECK: for (int i0_2 = 0; i0_2 < 5
        # CHECK-NEXT: for (int i1_2 = 0; i1_2 < 3
        # CHECK-NEXT: aten_softmax_exp
        # CHECK: for (int i${other_dim}_3 = 0; i${other_dim}_3 < ${other_dim_size}
        # CHECK: for (int i${softmax_dim}_3 = 0; i${softmax_dim}_3 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_sum
        # CHECK: for (int i0_4 = 0; i0_4 < 5
        # CHECK-NEXT: for (int i1_4 = 0; i1_4 < 3
        # CHECK-NEXT: aten_softmax)IR";

  for (int softmax_dim = 0; softmax_dim < a.dim(); ++softmax_dim) {
    auto softmax_dim_size = a.sizes()[softmax_dim];
    auto other_dim = (softmax_dim + 1) % a.dim();

    KernelScope kernel_scope;
    TemplateEnv env;
    env.d("dim", softmax_dim);
    const auto graph_string = format(graph_template, env);

    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    TemplateEnv ver_env;
    ver_env.d("other_dim", other_dim);
    ver_env.d("other_dim_size", a.sizes()[other_dim]);
    ver_env.d("softmax_dim", softmax_dim);
    ver_env.d("softmax_dim_size", softmax_dim_size);
    const auto verification_pattern = format(verification_template, ver_env);
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    auto output = stack[0].toTensor();
    auto ref = a.softmax(softmax_dim);
    ASSERT_EQ(output.sizes(), ref.sizes());
    ASSERT_TRUE(at::allclose(output, ref));
  }
}

void testKernelSoftmax3D() {
  const auto graph_template = R"IR(
      graph(%0 : Float(3, 4, 5, strides=[20, 5, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %2 : int = prim::Constant[value=7]()
        %3 : Tensor = aten::softmax(%0, %1, %2)
        return (%3))IR";

  auto a = at::rand({3, 4, 5}, TensorOptions(kCPU).dtype(at::kFloat));

  const std::string& verification_template =
      R"IR(
        # CHECK: for (int i0 = 0; i0 < 3
        # CHECK-NEXT: for (int i1 = 0; i1 < 4
        # CHECK-NEXT: for (int i2 = 0; i2 < 5
        # CHECK-NEXT: input1
        # CHECK: for (int i${dim1}_1 = 0; i${dim1}_1 < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2}_1 = 0; i${dim2}_1 < ${dim2_size}
        # CHECK: for (int i${softmax_dim}_1 = 0; i${softmax_dim}_1 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_max
        # CHECK: for (int i0_2 = 0; i0_2 < 3
        # CHECK-NEXT: for (int i1_2 = 0; i1_2 < 4
        # CHECK-NEXT: for (int i2_2 = 0; i2_2 < 5
        # CHECK-NEXT: aten_softmax_exp
        # CHECK: for (int i${dim1}_3 = 0; i${dim1}_3 < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2}_3 = 0; i${dim2}_3 < ${dim2_size}
        # CHECK: for (int i${softmax_dim}_3 = 0; i${softmax_dim}_3 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_sum
        # CHECK: for (int i0_4 = 0; i0_4 < 3
        # CHECK-NEXT: for (int i1_4 = 0; i1_4 < 4
        # CHECK-NEXT: for (int i2_4 = 0; i2_4 < 5
        # CHECK-NEXT: aten_softmax)IR";

  for (int softmax_dim = 0; softmax_dim < a.dim(); ++softmax_dim) {
    auto softmax_dim_size = a.sizes()[softmax_dim];
    std::vector<int> other_dims;
    for (int i = 0; i < a.dim(); ++i) {
      if (i != softmax_dim) {
        other_dims.push_back(i);
      }
    }

    KernelScope kernel_scope;
    TemplateEnv env;
    env.d("dim", softmax_dim);
    const auto graph_string = format(graph_template, env);

    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    TemplateEnv ver_env;
    ver_env.d("dim1", other_dims[0]);
    ver_env.d("dim1_size", a.sizes()[other_dims[0]]);
    ver_env.d("dim2", other_dims[1]);
    ver_env.d("dim2_size", a.sizes()[other_dims[1]]);
    ver_env.d("softmax_dim", softmax_dim);
    ver_env.d("softmax_dim_size", softmax_dim_size);
    const auto verification_pattern = format(verification_template, ver_env);
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    auto output = stack[0].toTensor();

    auto ref = a.softmax(softmax_dim);
    ASSERT_EQ(output.sizes(), ref.sizes());
    ASSERT_TRUE(at::allclose(output, ref));
  }
}

void testKernelSoftmax4D() {
  const auto graph_template = R"IR(
      graph(%0 : Float(2, 3, 2, 3, strides=[18, 6, 3, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %2 : int = prim::Constant[value=7]()
        %3 : Tensor = aten::softmax(%0, %1, %2)
        return (%3))IR";

  auto a = at::rand({2, 3, 2, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  const std::string& verification_template =
      R"IR(
        # CHECK: for (int i0 = 0; i0 < 2
        # CHECK-NEXT: for (int i1 = 0; i1 < 3
        # CHECK-NEXT: for (int i2 = 0; i2 < 2
        # CHECK-NEXT: for (int i3 = 0; i3 < 3
        # CHECK-NEXT: input1
        # CHECK: for (int i${dim1}_1 = 0; i${dim1}_1 < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2}_1 = 0; i${dim2}_1 < ${dim2_size}
        # CHECK-NEXT: for (int i${dim3}_1 = 0; i${dim3}_1 < ${dim3_size}
        # CHECK: for (int i${softmax_dim}_1 = 0; i${softmax_dim}_1 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_max
        # CHECK: for (int i0_2 = 0; i0_2 < 2
        # CHECK-NEXT: for (int i1_2 = 0; i1_2 < 3
        # CHECK-NEXT: for (int i2_2 = 0; i2_2 < 2
        # CHECK-NEXT: for (int i3_2 = 0; i3_2 < 3
        # CHECK-NEXT: aten_softmax_exp
        # CHECK: for (int i${dim1}_3 = 0; i${dim1}_3 < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2}_3 = 0; i${dim2}_3 < ${dim2_size}
        # CHECK-NEXT: for (int i${dim3}_3 = 0; i${dim3}_3 < ${dim3_size}
        # CHECK: for (int i${softmax_dim}_3 = 0; i${softmax_dim}_3 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_sum
        # CHECK: for (int i0_4 = 0; i0_4 < 2
        # CHECK-NEXT: for (int i1_4 = 0; i1_4 < 3
        # CHECK-NEXT: for (int i2_4 = 0; i2_4 < 2
        # CHECK-NEXT: for (int i3_4 = 0; i3_4 < 3
        # CHECK-NEXT: aten_softmax)IR";

  for (int softmax_dim = 0; softmax_dim < a.dim(); ++softmax_dim) {
    auto softmax_dim_size = a.sizes()[softmax_dim];
    std::vector<int> other_dims;
    for (int i = 0; i < a.dim(); ++i) {
      if (i != softmax_dim) {
        other_dims.push_back(i);
      }
    }

    KernelScope kernel_scope;
    TemplateEnv env;
    env.d("dim", softmax_dim);
    const auto graph_string = format(graph_template, env);

    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    TensorExprKernel k(graph);
    std::vector<at::Tensor> inputs = {a};
    Stmt* s = k.getCodeGenStmt();

    std::ostringstream oss;
    oss << *s;

    TemplateEnv ver_env;
    ver_env.d("dim1", other_dims[0]);
    ver_env.d("dim1_size", a.sizes()[other_dims[0]]);
    ver_env.d("dim2", other_dims[1]);
    ver_env.d("dim2_size", a.sizes()[other_dims[1]]);
    ver_env.d("dim3", other_dims[2]);
    ver_env.d("dim3_size", a.sizes()[other_dims[2]]);
    ver_env.d("softmax_dim", softmax_dim);
    ver_env.d("softmax_dim_size", softmax_dim_size);
    const auto verification_pattern = format(verification_template, ver_env);
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    auto output = stack[0].toTensor();
    auto ref = a.softmax(softmax_dim);
    ASSERT_EQ(output.sizes(), ref.sizes());
    ASSERT_TRUE(at::allclose(output, ref));
  }
}

namespace {

std::vector<int64_t> bufferSizes(const Placeholder& t) {
  std::vector<int64_t> sizes;
  sizes.reserve(t.ndim());
  for (int i = 0; i < t.ndim(); i++) {
    sizes.push_back(immediateAs<int64_t>(t.dim(i)));
  }
  return sizes;
}

Tensor* AtenBinary(
    const std::string& name,
    Tensor* lhs,
    const Placeholder& rhs,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        body) {
  return Compute(
      name,
      c10::fmap<DimArg>(c10::fmap<ExprHandle>(lhs->dims())),
      [&](const std::vector<VarHandle>& axes) {
        return body(lhs->call(axes), rhs.load(axes));
      });
}

Tensor* AtenCat(const std::vector<Placeholder>& inputs, size_t dim) {
  int concat_dim_size = 0;
  for (const auto& x : inputs) {
    concat_dim_size += immediateAs<int>(x.dim(dim));
  }
  std::vector<int> concat_sizes;
  for (const auto dim : inputs.front().dims()) {
    concat_sizes.push_back(immediateAs<int>(dim));
  }
  concat_sizes[dim] = concat_dim_size;
  std::vector<DimArg> output_dims;
  for (const auto size : concat_sizes) {
    output_dims.emplace_back(ExprHandle(size));
  }
  return Compute(
      "aten_cat", output_dims, [&](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> newAxes(axes.begin(), axes.end());
        ExprHandle load = inputs[0].load(axes);
        size_t offset = bufferSizes(inputs[0])[dim];
        newAxes[dim] = newAxes[dim] - IntImm::make(offset);

        for (size_t ii = 1; ii < inputs.size(); ++ii) {
          load = ifThenElse(
              CompareSelect::make(axes[dim], IntImm::make(offset), kLT),
              load,
              inputs[ii].load(newAxes));
          offset += bufferSizes(inputs[ii])[dim];
          newAxes[dim] = axes[dim] - IntImm::make(offset);
        }
        return load;
      });
}

Tensor* ReplaceNanWithZero(Tensor* x) {
  return Compute(
      "replace_nans_with_zero",
      c10::fmap<DimArg>(c10::fmap<ExprHandle>(x->dims())),
      [&](const std::vector<VarHandle>& axes) {
        return ifThenElse(
            x->call(axes) == x->call(axes),
            x->call(axes),
            ExprHandle(float(0)));
      });
}

Tensor* AtenClamp(Tensor* x, const float min_val, const float max_val) {
  return Compute(
      "clamp",
      c10::fmap<DimArg>(c10::fmap<ExprHandle>(x->dims())),
      [&](const std::vector<VarHandle>& axes) {
        auto in = x->call(axes);
        ExprHandle min(min_val);
        ExprHandle max(max_val);
        return CompareSelect::make(
            in, min, min, CompareSelect::make(in, max, max, in, kGT), kLT);
      });
}

} // namespace

// clang-format off
// test_tensorexpr --gtest_filter=TensorExprTest.KernelConcatAddMulReplaceNanClip
// clang-format on
void testKernelConcatAddMulReplaceNanClip() {
  KernelScope kernel_scope;
  std::vector<Placeholder> inps;
  int b = 1;
  int d = 10;
  int concat_size = 0;
  std::vector<ExprHandle> in_sizes;
  in_sizes.emplace_back(b);
  in_sizes.emplace_back(d);
  std::vector<BufHandle> in_bufs;
  for (int i = 0; i < 4; ++i) {
    in_bufs.emplace_back("input", in_sizes, kFloat);
    concat_size += d;
  }
  for (const auto& in_buf : in_bufs) {
    inps.emplace_back(in_buf);
  }
  std::vector<ExprHandle> sizes;
  sizes.emplace_back(b);
  sizes.emplace_back(concat_size);
  BufHandle add_in_buf("add_in", sizes, kFloat);
  Placeholder add_in(add_in_buf);
  BufHandle mul_in_buf("mul_in", sizes, kFloat);
  Placeholder mul_in(mul_in_buf);
  auto cat = AtenCat(inps, 1);
  auto add = AtenBinary(
      "add", cat, add_in, [](const ExprHandle& x, const ExprHandle& y) {
        return x + y;
      });
  auto mul = AtenBinary(
      "mul", add, mul_in, [](const ExprHandle& x, const ExprHandle& y) {
        return x * y;
      });
  auto no_nans = ReplaceNanWithZero(mul);
  auto clamp = AtenClamp(no_nans, -10.0, 10.0);
  LoopNest l({cat, mul, no_nans, clamp});
  l.computeInline(l.getLoopBodyFor(add));
  l.prepareForCodegen();
  Stmt* stmt = l.root_stmt();
  stmt = IRSimplifier::simplify(stmt);
  std::vector<torch::jit::tensorexpr::CodeGen::BufferArg> formal_parameters;
  for (const auto in : inps) {
    formal_parameters.emplace_back(in);
  }
  formal_parameters.emplace_back(add_in);
  formal_parameters.emplace_back(mul_in);
  formal_parameters.emplace_back(cat);
  formal_parameters.emplace_back(mul);
  formal_parameters.emplace_back(no_nans);
  formal_parameters.emplace_back(clamp);
  // std::cerr << *stmt << "\n";
  auto codegen = CreateCodeGen("llvm_codegen", stmt, formal_parameters);
  // Call the generated kernel.
  std::vector<at::Tensor> arg_tensors;
  for (const auto in : inps) {
    arg_tensors.push_back(
        at::randn(bufferSizes(in), at::TensorOptions(at::kFloat)));
  }
  arg_tensors.push_back(
      at::randn(bufferSizes(add_in), at::TensorOptions(at::kFloat)));
  arg_tensors.push_back(
      at::randn(bufferSizes(mul_in), at::TensorOptions(at::kFloat)));
  arg_tensors.push_back(
      at::randn(bufferSizes(cat), at::TensorOptions(at::kFloat)));
  arg_tensors.push_back(arg_tensors.back());
  arg_tensors.push_back(arg_tensors.back());
  arg_tensors.push_back(arg_tensors.back());
  std::vector<torch::jit::tensorexpr::CodeGen::CallArg> args;
  for (const auto& arg_tensor : arg_tensors) {
    args.emplace_back(arg_tensor.data_ptr());
  }
  codegen->call(args);
  //  Call the reference implementation.
  auto ref_cat = at::cat(
      std::vector<at::Tensor>(
          arg_tensors.begin(), arg_tensors.begin() + inps.size()),
      1);
  auto ref_add = ref_cat + arg_tensors[inps.size()];
  auto ref_mul = ref_add * arg_tensors[inps.size() + 1];
  at::index_put_(
      ref_mul,
      at::isnan(ref_mul),
      at::scalar_tensor(0, at::TensorOptions(at::kFloat)));
  auto ref_clamp = at::clamp(ref_mul, -10.0, 10.0);
  ASSERT_TRUE(ref_clamp.allclose(arg_tensors.back()));
}

} // namespace jit
} // namespace torch
