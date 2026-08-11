import torch

from heston_fno.evaluation import ErrorSummary, absolute_error, mean_absolute_error, mean_relative_error, rmse, summarize_errors


def test_error_metrics_and_summary() -> None:
	prediction = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
	target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

	assert torch.equal(absolute_error(prediction, target), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
	assert torch.isclose(mean_absolute_error(prediction, target), torch.tensor(2.5))
	assert torch.isclose(mean_relative_error(prediction, target), torch.tensor(((1 / 1) + (2 / 2) + (3 / 3) + (4 / 4)) / 4))
	assert torch.isclose(rmse(prediction, target), torch.sqrt(torch.tensor(7.5)))

	summary = summarize_errors(prediction, target)
	assert isinstance(summary, ErrorSummary)
	assert set(summary.to_dict()) == {
		"mean_absolute_error",
		"mean_relative_error",
		"rmse",
		"max_absolute_error",
	}
