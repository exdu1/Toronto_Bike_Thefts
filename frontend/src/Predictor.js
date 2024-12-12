import React, { useState } from 'react';
import axios from 'axios';

const Predictor = () => {
  const [features, setFeatures] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFeatures({
      ...features,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5001/predict', {
        features: Object.values(features).map(Number),
      });
      setPrediction(response.data.prediction);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div>
      <h2>Bicycle Theft Prediction</h2>
      <form onSubmit={handleSubmit}>
        <input type="number" name="OBJECTID" placeholder="Enter OBJECTID" onChange={handleInputChange} />
        <input type="number" name="EVENT_UNIQUE_ID" placeholder="Enter EVENT_UNIQUE_ID" onChange={handleInputChange} />
        <input type="number" name="PRIMARY_OFFENCE" placeholder="Enter PRIMARY_OFFENCE" onChange={handleInputChange} />
        <input type="number" name="OCC_DATE" placeholder="Enter OCC_DATE" onChange={handleInputChange} />
        <input type="number" name="OCC_YEAR" placeholder="Enter OCC_YEAR" onChange={handleInputChange} />
        <input type="number" name="OCC_MONTH" placeholder="Enter OCC_MONTH" onChange={handleInputChange} />
        <input type="number" name="OCC_DOW" placeholder="Enter OCC_DOW" onChange={handleInputChange} />
        <input type="number" name="OCC_DAY" placeholder="Enter OCC_DAY" onChange={handleInputChange} />
        <input type="number" name="OCC_DOY" placeholder="Enter OCC_DOY" onChange={handleInputChange} />
        <input type="number" name="OCC_HOUR" placeholder="Enter OCC_HOUR" onChange={handleInputChange} />
        <input type="number" name="REPORT_DATE" placeholder="Enter REPORT_DATE" onChange={handleInputChange} />
        <input type="number" name="REPORT_YEAR" placeholder="Enter REPORT_YEAR" onChange={handleInputChange} />
        <input type="number" name="REPORT_MONTH" placeholder="Enter REPORT_MONTH" onChange={handleInputChange} />
        <input type="number" name="REPORT_DOW" placeholder="Enter REPORT_DOW" onChange={handleInputChange} />
        <input type="number" name="REPORT_DAY" placeholder="Enter REPORT_DAY" onChange={handleInputChange} />
        <input type="number" name="REPORT_DOY" placeholder="Enter REPORT_DOY" onChange={handleInputChange} />
        <input type="number" name="REPORT_HOUR" placeholder="Enter REPORT_HOUR" onChange={handleInputChange} />
        <input type="number" name="DIVISION" placeholder="Enter DIVISION" onChange={handleInputChange} />
        <input type="number" name="LOCATION_TYPE" placeholder="Enter LOCATION_TYPE" onChange={handleInputChange} />
        <input type="number" name="PREMISES_TYPE" placeholder="Enter PREMISES_TYPE" onChange={handleInputChange} />
        <input type="number" name="BIKE_MAKE" placeholder="Enter BIKE_MAKE" onChange={handleInputChange} />
        <input type="number" name="BIKE_MODEL" placeholder="Enter BIKE_MODEL" onChange={handleInputChange} />
        <input type="number" name="BIKE_TYPE" placeholder="Enter BIKE_TYPE" onChange={handleInputChange} />
        <input type="number" name="BIKE_SPEED" placeholder="Enter BIKE_SPEED" onChange={handleInputChange} />
        <input type="number" name="BIKE_COLOUR" placeholder="Enter BIKE_COLOUR" onChange={handleInputChange} />
        <input type="number" name="BIKE_COST" placeholder="Enter BIKE_COST" onChange={handleInputChange} />
        <input type="number" name="HOOD_158" placeholder="Enter HOOD_158" onChange={handleInputChange} />
        <input type="number" name="NEIGHBOURHOOD_158" placeholder="Enter NEIGHBOURHOOD_158" onChange={handleInputChange} />
        <input type="number" name="HOOD_140" placeholder="Enter HOOD_140" onChange={handleInputChange} />
        <input type="number" name="NEIGHBOURHOOD_140" placeholder="Enter NEIGHBOURHOOD_140" onChange={handleInputChange} />
        <input type="number" name="LONG_WGS84" placeholder="Enter LONG_WGS84" onChange={handleInputChange} />
        <input type="number" name="LAT_WGS84" placeholder="Enter LAT_WGS84" onChange={handleInputChange} />
        <input type="number" name="x" placeholder="Enter x" onChange={handleInputChange} />
        <input type="number" name="y" placeholder="Enter y" onChange={handleInputChange} />
        <button type="submit">Predict</button>
      </form>
      {prediction !== null && <p>Prediction: {prediction}</p>}
      {error && <p>Error: {error}</p>}
    </div>
  );
};

export default Predictor;
