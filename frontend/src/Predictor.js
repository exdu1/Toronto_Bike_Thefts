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
    <div className="container mt-5">
      <h2 className="mb-4">Bicycle Theft Prediction</h2>
      <form onSubmit={handleSubmit}>
        <div className="mb-3">
          <label htmlFor="OBJECTID" className="form-label">OBJECTID</label>
          <input
            type="number"
            name="OBJECTID"
            className="form-control"
            id="OBJECTID"
            placeholder="Enter OBJECTID"
            onChange={handleInputChange}
          />
        </div>
        <div className="mb-3">
          <label htmlFor="EVENT_UNIQUE_ID" className="form-label">EVENT_UNIQUE_ID</label>
          <input
            type="number"
            name="EVENT_UNIQUE_ID"
            className="form-control"
            id="EVENT_UNIQUE_ID"
            placeholder="Enter EVENT_UNIQUE_ID"
            onChange={handleInputChange}
          />
        </div>
        <div className="mb-3">
          <label htmlFor="PRIMARY_OFFENCE" className="form-label">PRIMARY_OFFENCE</label>
          <input
            type="number"
            name="PRIMARY_OFFENCE"
            className="form-control"
            id="PRIMARY_OFFENCE"
            placeholder="Enter PRIMARY_OFFENCE"
            onChange={handleInputChange}
          />
        </div>
        {/* Repeat the above for all remaining inputs */}
        <div className="mb-3">
          <label htmlFor="OCC_DATE" className="form-label">OCC_DATE</label>
          <input
            type="number"
            name="OCC_DATE"
            className="form-control"
            id="OCC_DATE"
            placeholder="Enter OCC_DATE"
            onChange={handleInputChange}
          />
        </div>
        {/* ... Add remaining input fields in similar fashion ... */}
        <button type="submit" className="btn btn-primary">Predict</button>
      </form>
      {prediction !== null && (
        <div className="alert alert-success mt-4">
          <strong>Prediction: </strong> {prediction}
        </div>
      )}
      {error && (
        <div className="alert alert-danger mt-4">
          <strong>Error: </strong> {error}
        </div>
      )}
    </div>
  );
};

export default Predictor;
