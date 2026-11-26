"""Independent, fold-safe supervised evaluation for Mathy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import throw_if


def create_predictive_pipeline( df_features: pd.DataFrame, estimator: object ) -> Pipeline:
	"""Create an unfitted preprocessing and estimator pipeline.

	Purpose:
	    Fits numeric and categorical transformations only on the rows passed to fit.

	Args:
	    df_features (pd.DataFrame): Feature schema used to select column types.
	    estimator (object): Unfitted scikit-learn estimator.

	Returns:
	    Pipeline: Cloneable preprocessing and estimation pipeline.
	"""
	throw_if( 'df_features', df_features )
	throw_if( 'estimator', estimator )
	if not isinstance( df_features, pd.DataFrame ) or df_features.empty:
		raise ValueError( 'Predictors must be a nonempty dataframe.' )
	numeric = df_features.select_dtypes( include='number' ).columns.tolist( )
	categorical = [ column for column in df_features.columns if column not in numeric ]
	transformers = [ ]
	if numeric:
		transformers.append( ( 'numeric', Pipeline( [
			( 'imputer', SimpleImputer( strategy='median', keep_empty_features=True ) ),
			( 'scaler', StandardScaler( ) ) ] ), numeric ) )
	if categorical:
		transformers.append( ( 'categorical', Pipeline( [
			( 'imputer', SimpleImputer( strategy='most_frequent', keep_empty_features=True ) ),
			( 'encoder', OneHotEncoder( handle_unknown='ignore', sparse_output=False ) ) ] ), categorical ) )
	return Pipeline( [ ( 'preprocessing', ColumnTransformer( transformers ) ),
		( 'estimator', estimator ) ] )


def evaluate_predictive_model( df_source: pd.DataFrame, features: list[ str ], target: str,
	mode: str, name: str, size: float=0.2, random: int=42, folds: int=5 ) -> pd.DataFrame:
	"""Evaluate a supervised model with independent holdout and cross-validation.

	Purpose:
	    Fits transformations within each training fold, including stacking base estimators.
	    The held-out test partition is excluded from cross-validation and fitting.

	Args:
	    df_source (pd.DataFrame): Unprocessed source dataset.
	    features (list[str]): Unique predictor columns.
	    target (str): Target column excluded from predictors.
	    mode (str): Classification or regression.
	    name (str): Supported estimator name.
	    size (float): Held-out fraction strictly between zero and one.
	    random (int): Nonnegative 32-bit random seed.
	    folds (int): Number of cross-validation partitions, at least two.

	Returns:
	    pd.DataFrame: Holdout score, cross-validation scores and row counts.

	Raises:
	    ValueError: Invalid data, estimator configuration or partition sizes.
	"""
	for label, value in [ ( 'df_source', df_source ), ( 'features', features ),
		( 'target', target ), ( 'mode', mode ), ( 'name', name ) ]:
		throw_if( label, value )
	if not isinstance( df_source, pd.DataFrame ) or not df_source.columns.is_unique:
		raise ValueError( 'Source data must have unique columns.' )
	if not isinstance( features, list ) or not features or len( set( features ) ) != len( features ):
		raise ValueError( 'Select unique predictor columns.' )
	if target in features or any( column not in df_source.columns for column in features + [ target ] ):
		raise ValueError( 'Predictors and target must be distinct existing columns.' )
	if mode not in ( 'classification', 'regression' ):
		raise ValueError( 'Unsupported predictive mode.' )
	if isinstance( size, bool ) or not isinstance( size, ( int, float ) ) or not 0 < size < 1:
		raise ValueError( 'Test size must be between zero and one.' )
	if type( random ) is not int or not 0 <= random <= 4294967295:
		raise ValueError( 'Invalid random seed.' )
	if type( folds ) is not int or folds < 2:
		raise ValueError( 'Cross-validation requires at least two folds.' )
	classification = mode == 'classification'
	df_model = df_source[ features + [ target ] ].copy( )
	if not classification:
		df_model[ target ] = pd.to_numeric( df_model[ target ], errors='raise' ).replace( [ np.inf, -np.inf ], np.nan )
	df_model = df_model.dropna( subset=[ target ] )
	X = df_model[ features ].copy( )
	y = df_model[ target ]
	for column in X.columns:
		if pd.api.types.is_numeric_dtype( X[ column ] ) and not pd.api.types.is_bool_dtype( X[ column ] ):
			X[ column ] = X[ column ].replace( [ np.inf, -np.inf ], np.nan ).astype( float )
		else:
			X[ column ] = X[ column ].map( lambda value: str( value ) if pd.notna( value ) else np.nan ).astype( object )
	if classification:
		y = y.astype( str )
	if y.nunique( ) < 2:
		raise ValueError( 'The target must contain at least two distinct values.' )
	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=size,
		random_state=random, stratify=y if classification else None )
	if classification:
		if y_train.value_counts( ).min( ) < folds:
			raise ValueError( 'Each training class must have at least as many observations as folds.' )
		cv = StratifiedKFold( n_splits=folds, shuffle=True, random_state=random )
		scoring = 'accuracy'
		linear = LogisticRegression( max_iter=2000, random_state=random )
		forest = RandomForestClassifier( random_state=random )
		models = { 'Logistic Regression': linear, 'Random Forest': forest }
	else:
		if len( X_test ) < 2 or len( X_train ) < folds * 2:
			raise ValueError( 'Regression requires at least two observations in each evaluation partition.' )
		cv = KFold( n_splits=folds, shuffle=True, random_state=random )
		scoring = 'r2'
		linear = LinearRegression( )
		forest = RandomForestRegressor( random_state=random )
		models = { 'Ordinary Least Squares': linear, 'Random Forest': forest }
	if name == 'Stacking':
		# Each base pipeline is cloned and fitted inside stacking's internal folds.
		# The outer cross-validation also clones the complete ensemble.
		for train_index, validation_index in cv.split( X_train, y_train ):
			if classification and y_train.iloc[ train_index ].value_counts( ).min( ) < 2:
				raise ValueError( 'Stacking requires at least two observations per class.' )
			if not classification and len( train_index ) < 2:
				raise ValueError( 'Stacking requires at least two observations in every outer training fold.' )
		estimators = [ ( 'linear', create_predictive_pipeline( X_train, linear ) ),
			( 'forest', create_predictive_pipeline( X_train, forest ) ) ]
		model = StackingClassifier( estimators=estimators, cv=2 ) if classification else StackingRegressor( estimators=estimators, final_estimator=Ridge( ), cv=2 )
	elif name in models:
		model = create_predictive_pipeline( X_train, models[ name ] )
	else:
		raise ValueError( 'Unsupported predictive estimator.' )
	scores = cross_val_score( model, X_train, y_train, cv=cv, scoring=scoring, error_score='raise' )
	model.fit( X_train, y_train )
	predictions = model.predict( X_test )
	holdout = accuracy_score( y_test, predictions ) if classification else r2_score( y_test, predictions )
	return pd.DataFrame( { 'Metric': [ 'Holdout ' + scoring, 'Cross-validation mean ' + scoring,
		'Cross-validation standard deviation', 'Training rows', 'Testing rows' ],
		'Value': [ holdout, scores.mean( ), scores.std( ), len( X_train ), len( X_test ) ] } )
