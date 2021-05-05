from mysklearn.myclassifiers import MyRandomForestClassifier

interview_table = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

interview_class_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

def test_my_random_forest_fit():
    interview_classifier = MyRandomForestClassifier()
    interview_classifier.fit(interview_table, interview_class_train, 2, 20, 7, 3)
    forest = [['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 0, 7]], ['Value', 'yes', ['Leaf', 'True', 2, 9]]], ['Leaf', 'True', 0, 9], ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 2, 4]], ['Value', 'yes', ['Leaf', 'False', 2, 4]]]], ['Value', 'Mid', ['Leaf', 'True', 4, 9]], ['Value', 'Senior', ['Leaf', 'False', 1, 9]]], ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 2, 4]], ['Value', 'yes', ['Leaf', 'False', 2, 4]]]], ['Value', 'Mid', ['Leaf', 'True', 4, 9]], ['Value', 'Senior', ['Leaf', 'False', 1, 9]]], ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 2, 4]], ['Value', 'yes', ['Leaf', 'False', 2, 4]]]], ['Value', 'Mid', ['Leaf', 'True', 4, 9]], ['Value', 'Senior', ['Leaf', 'False', 1, 9]]], ['Leaf', 'True', 0, 9], ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 2, 4]], ['Value', 'yes', ['Leaf', 'False', 2, 4]]]], ['Value', 'Mid', ['Leaf', 'True', 4, 9]], ['Value', 'Senior', ['Leaf', 'False', 1, 9]]]]
    assert [i for i in interview_classifier.forest if i not in forest] == []

def test_decision_tree_classifier_predict():
    interview_classifier = MyRandomForestClassifier()
    interview_classifier.fit(interview_table, interview_class_train, 2, 20, 7, 3)
    assert interview_classifier.predict([["Mid", "Java", "yes", "no"], ["Junior", "Python", "no", "yes"]]) == ["True", "False"]
  