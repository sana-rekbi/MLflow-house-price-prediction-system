import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Here, we simulate importing or generating some data.
    # In a real-world scenario, this could be an API call, database query, or loading from a file.
    data = {
        "Order": [1, 2],
        "PID": [526301100, 526301101],
        "MS SubClass": [60, 20],
        "Lot Frontage": [65.0, 80.0],
        "Lot Area": [8450, 9600],
        "Overall Qual": [7, 6],
        "Overall Cond": [5, 8],
        "Year Built": [2003, 1976],
        "Year Remod/Add": [2003, 1976],
        "Mas Vnr Area": [196.0, 0.0],
        "BsmtFin SF 1": [706, 978],
        "BsmtFin SF 2": [0, 0],
        "Bsmt Unf SF": [150, 284],
        "Total Bsmt SF": [856, 1262],
        "1st Flr SF": [856, 1262],
        "2nd Flr SF": [854, 0],
        "Low Qual Fin SF": [0, 0],
        "Gr Liv Area": [1710, 1262],
        "Bsmt Full Bath": [1, 0],
        "Bsmt Half Bath": [0, 1],
        "Full Bath": [2, 2],
        "Half Bath": [1, 0],
        "Bedroom AbvGr": [3, 3],
        "Kitchen AbvGr": [1, 1],
        "TotRms AbvGrd": [8, 6],
        "Fireplaces": [0, 1],
        "Garage Yr Blt": [2003, 1976],
        "Garage Cars": [2, 2],
        "Garage Area": [548, 460],
        "Wood Deck SF": [0, 298],
        "Open Porch SF": [61, 0],
        "Enclosed Porch": [0, 0],
        "3Ssn Porch": [0, 0],
        "Screen Porch": [0, 0],
        "Pool Area": [0, 0],
        "Misc Val": [0, 0],
        "Mo Sold": [2, 5],
        "Yr Sold": [2008, 2007],
    }

    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data
