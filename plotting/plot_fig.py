import plotly.graph_objects as go
from plotly.subplots import make_subplots

epsilons = [1, 2, 5, 10, 20, 30]
sigmas = [3.86718, 2.13867, 1.12548, 0.80230, 0.62690, 0.62995]
train_losses = [2.7874, 0.7625, 0.4183, 0.3590, 0.3327, 0.2906]
train_accs = [83.46, 87.85, 90.85, 91.27, 91.70, 93.22]
test_losses = [2.6155, 0.7388, 0.3925, 0.3448, 0.3241, 0.3021]
test_accs = [83.28, 88.59, 91.50, 92.12, 92.54, 93.32]

# Create figure with secondary y-axis
# fig = make_subplots(specs=[[{"secondary_y": True}]])
fig = make_subplots()
# Add traces
fig.add_trace(
    go.Scatter(x=epsilons, y=sigmas, name="Sigma")
)

# fig.add_trace(
#     go.Scatter(x=epsilons, y=test_accs, name="Test Accuracy"),
#     secondary_y=False,
# )

# fig.add_trace(
#     go.Scatter(x=epsilons, y=train_losses, name="Train Loss"),
#     secondary_y=True,
# )

# fig.add_trace(
#     go.Scatter(x=epsilons, y=test_losses, name="Test Loss"),
#     secondary_y=True,
# )

# Add figure title
fig.update_layout(
    title_text="Epsilon vs Sigma"
)

# Set x-axis title
fig.update_xaxes(title_text="Epsilon")

# Set y-axes titles
fig.update_yaxes(title_text="Sigma")
# fig.update_yaxes(title_text="Loss", secondary_y=True)

fig.show()