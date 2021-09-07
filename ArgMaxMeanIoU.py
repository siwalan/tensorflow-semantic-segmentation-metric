import tensorflow as tf

class  ArgMaxMeanIoU(tf.keras.metrics.Metric):
    ## Based on TF Basic MeanIoU Function
    def __init__(self, num_classes, labels=None, dtype=None):
        super(ArgMaxMeanIoU, self).__init__(dtype=dtype)
        self.num_classes = num_classes

        # Variable to accumulate the predictions in the confusion matrix.
        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros')        
        
        if labels != None:
            self.labels = labels
        
    def update_state(self, y_true, y_pred, threshold = None, sample_weight=None):
        y_true = tf.cast(y_true, dtype =self._dtype) ### To follow the TF Standard...
        y_true = tf.math.argmax(y_true,axis=0)

        y_pred = tf.cast(y_pred, dtype = self._dtype)
        y_pred = tf.math.argmax(y_pred,axis=0)

        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])
        
        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true,[-1])
            
        current_cm = tf.math.confusion_matrix(
            y_true, y_pred,
            self.num_classes, dtype=self._dtype)
        
        return self.total_cm.assign_add(current_cm)
    
    def result(self):
        sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0),dtype = self._dtype)
        sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1),dtype = self._dtype)
        true_positives = tf.cast(tf.linalg.tensor_diag_part(self.total_cm),dtype = self._dtype)
        
        denominator = sum_over_row + sum_over_col - true_positives 

        num_valid_entries = tf.reduce_sum(tf.cast(tf.not_equal(denominator,0),dtype=self._dtype))
        
        iou = tf.math.divide_no_nan(true_positives,denominator)
        return tf.math.divide_no_nan(tf.reduce_sum(iou, name='mean_iou'),num_valid_entries)
                                                  
    def reset_state(self):
        tf.keras.backend.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))                        
               
    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(ArgMaxMeanIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
