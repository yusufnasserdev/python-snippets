# create pipeline
pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                   ('model', LinearRegression())])

# create grid search object
param_grid = {'model__fit_intercept':[True, False]}
gs = GridSearchCV(estimator=pipeline,
                  param_grid=param_grid,
                 scoring='neg_mean_squared_error', 
                 cv=3, 
                 n_jobs=-1)

# perform grid search
gs.fit(X_train, y_train)