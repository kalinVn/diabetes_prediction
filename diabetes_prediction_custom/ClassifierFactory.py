from sklearn import svm


class ClassifierFactory:

    def get_classifier_by_name(self, name):
        if name == "svm":
            return svm.SVC(kernel='linear')



