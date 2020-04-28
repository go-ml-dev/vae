package tests

import (
	"fmt"
	"go-ml.dev/pkg/base/fu"
	"go-ml.dev/pkg/base/model"
	"go-ml.dev/pkg/dataset/mnist"
	"go-ml.dev/pkg/iokit"
	"go-ml.dev/pkg/nn"
	"go-ml.dev/pkg/nn/mx"
	"go-ml.dev/pkg/vae"
	"gotest.tools/assert"
	"testing"
)

func Test_mnistVae(t *testing.T) {
	modelFile := iokit.File(fu.ModelPath("mnist_test_conv0.zip"))

	report := vae.Model{
		Optimizer: nn.Adam{Lr: .001},
		Seed:      43,
		Hidden:    128,
		Latent:    16,
		Context:   mx.GPU,
	}.Feed(model.Dataset{
		Source:   mnist.Data.RandomFlag("Test", 42, 0.2),
		Test:     "Test",
		Features: []string{"Image"},
	}).LuckyTrain(model.Training{
		Iterations: 8,
		ModelFile:  modelFile,
		Metrics:    model.Regression{Error: 0.019},
		Score:      model.LossScore,
	})

	fmt.Println(report.TheBest, report.Score)
	fmt.Println(report.History.Round(5))
	assert.Assert(t, model.Error(report.Test) < 0.019)

	x := mnist.T10k.Alias("Image", "Ilabel")
	pred1 := nn.LuckyObjectify(modelFile, vae.RecoderCollection)
	lr := model.LuckyEvaluate(x, "Ilabel", pred1, 32, &model.Regression{})
	fmt.Println(lr.Round(5))
	assert.Assert(t, model.Error(lr) < 0.019)
}
