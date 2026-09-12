import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from proj import (CATEGORICAL_FEATURES, RAW_NUMERIC, create_pipeline,
                  HousingFeatureEngineer, validate_schema, split_features_target)
from evaluate import split_indices, metrics


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep the public test suite independent from the course dataset.
        rows = 300
        data = pd.DataFrame(index=np.arange(rows))
        for column in RAW_NUMERIC:
            data[column] = np.arange(rows, dtype=float) % 17 + 1
        for column in CATEGORICAL_FEATURES:
            data[column] = np.arange(rows) % 5
        data['Sale Year'] = 2013 + np.arange(rows) % 7
        data['Modeling Group'] = 'A'
        data['PIN'] = [f'PIN-{i % 250:04d}' for i in range(rows)]
        data['Description'] = '8 rooms, 4 of which are bedrooms, and 2.5 of which are bathrooms.'
        data['Estimate (Land)'] = 75_000 + np.arange(rows) * 10
        data['Estimate (Building)'] = 175_000 + np.arange(rows) * 20
        data['Sale Price'] = 250_000 + np.arange(rows) * 100
        cls.data = data

    def test_group_split_has_no_pin_overlap(self):
        X = pd.concat([self.data,self.data],ignore_index=True)
        a,b = split_indices(X,'group')
        self.assertFalse(set(X.iloc[a].PIN)&set(X.iloc[b].PIN))
        self.assertFalse(set(a)&set(b))

    def test_time_split_is_purged(self):
        a,b = split_indices(self.data,'time')
        self.assertLess(self.data.iloc[a]['Sale Year'].max(),self.data.iloc[b]['Sale Year'].min())
        self.assertFalse(set(self.data.iloc[a].PIN)&set(self.data.iloc[b].PIN))

    def test_schema_fails_early(self):
        with self.assertRaisesRegex(ValueError,'Missing required'):
            validate_schema(self.data.drop(columns='Building Square Feet'))
        bad=self.data.copy(); bad['Building Square Feet']=np.inf
        with self.assertRaisesRegex(ValueError,'Invalid numeric'):
            validate_schema(bad)

    def test_features_keep_index_and_fractional_bathrooms(self):
        data=self.data.iloc[[11,8]].copy()
        data['Description']='12 rooms, 10 of which are bedrooms, and 3.5 of which are bathrooms.'
        before=data.copy(deep=True)
        features=HousingFeatureEngineer().fit_transform(data)
        pd.testing.assert_frame_equal(data,before)
        self.assertEqual(features.index.tolist(),[11,8])
        self.assertEqual(features['Bathrooms'].tolist(),[3.5,3.5])
        self.assertEqual(features['Bedrooms'].tolist(),[10,10])

    def test_no_estimates_unknown_categories_and_reload(self):
        data=self.data.drop(columns=['Estimate (Land)','Estimate (Building)'])
        X,y=split_features_target(data)
        model=create_pipeline(include_estimates=False).set_params(model__max_iter=5)
        model.fit(X,y)
        test=X.iloc[:3].copy(); test['Modeling Group']='UNSEEN'; test['Age']=np.nan
        result=model.predict(test)
        self.assertTrue(np.isfinite(result).all())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.joblib'; joblib.dump(model,path)
            np.testing.assert_allclose(joblib.load(path).predict(test),result)

    def test_invalid_targets_and_metrics(self):
        data=self.data.copy(); data.loc[data.index[:2],'Sale Price']=[0,-1]
        X,y=split_features_target(data)
        self.assertEqual(len(X),len(data)-2)
        self.assertEqual(metrics(y,y),dict(mae=0.,rmse=0.,log_rmse=0.))


if __name__=='__main__':
    unittest.main()
