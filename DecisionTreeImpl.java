import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.omg.CORBA.PUBLIC_MEMBER;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 * 
 * You must add code for the 5 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
	private DecTreeNode root;
	private List<String> labels; // ordered list of class labels
	private List<String> attributes; // ordered list of attributes
	private Map<String, List<String>> attributeValues; // map to ordered
														// discrete values taken
														// by attributes
	private DataSet train;

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary
		// this is void purposefully
	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train: the training set
	 */
	DecisionTreeImpl(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		this.train = train;
		// TODO: add code here
		List<Instance> instances = train.instances;
		ArrayList<Integer> attributeIDList = new ArrayList<Integer>();
		for (int k = 0; k < train.attributes.size(); k++){
			attributeIDList.add(k);
		}
		
		int bestAttributeID = getBestAttributeID(instances, attributeIDList);
		int label = this.getMajority(instances, train.labels.size());
		
		// if all the instances have the same label or there is no attribute, root will be the only node.
		// for the root, choose the best attribute. Then, remove it from the attributeIDList
		
		if (instances.size()>0){
			this.root = new DecTreeNode(label, bestAttributeID, -1, this.isDone(instances) || attributeIDList.size()==0);
			int index = attributeIDList.indexOf(bestAttributeID);
			attributeIDList.remove(index);
			this.buildTree(train.instances, attributeIDList, root);
		}
	}

	
	public void buildTree(List<Instance> instances, ArrayList<Integer> attributeIDList, DecTreeNode parent){
		
		//if parent is the terminal node, stop at parent
		if(parent.terminal) {
			return;
		}
		// else, create its children node, each label in the parent's attribute corresponds to one child
		int parentAttribute = parent.attribute;
		String attributeStr = train.attributes.get(parentAttribute);
		int numberOfChildren = train.attributeValues.get(attributeStr).size();
		
		
		ArrayList<ArrayList<Instance>> sortedByAttribute = new ArrayList<ArrayList<Instance>>();
		for (int k = 0; k < numberOfChildren ; k++){
			ArrayList<Instance> childInstances = new ArrayList<Instance>();
			sortedByAttribute.add(childInstances);			
		}
		
		for (int i = 0; i < instances.size(); i++){
			Instance instance = instances.get(i);
			int attributeLabel = instance.attributes.get(parentAttribute);
			sortedByAttribute.get(attributeLabel).add(instance);
		}
		// for each child, create the child node, add it to the parent. 
		for (int k = 0; k < numberOfChildren; k++){
			int childMajorityLabel;
			
			if (sortedByAttribute.get(k).size() > 0){
				childMajorityLabel = this.getMajority(sortedByAttribute.get(k), train.labels.size());
				int childAttribute = this.getBestAttributeID(sortedByAttribute.get(k), attributeIDList);
				boolean childTerminal = this.isDone(sortedByAttribute.get(k)) || attributeIDList.size() == 0;
				DecTreeNode childNode = new DecTreeNode(childMajorityLabel, childAttribute, k, childTerminal);
				parent.children.add(childNode);
			}
			else {
				childMajorityLabel = parent.label;
				DecTreeNode childNode = new DecTreeNode(childMajorityLabel, -1, k, true);
				parent.children.add(childNode);				
			}
		}

		for (int k = 0; k < parent.children.size(); k++) {
			DecTreeNode childrenNode = parent.children.get(k);
			int index = attributeIDList.indexOf(childrenNode.attribute);
			if (index >= 0) {
				attributeIDList.remove(index);
			}

			buildTree(sortedByAttribute.get(k), attributeIDList, childrenNode);
			
			if (index >= 0) {
				attributeIDList.add(childrenNode.attribute);
			}
		}
	}
	
	
	public boolean isDone(List<Instance> instances){
		boolean isDone = true;
		if (instances.size() == 0){
			return true;
		}
		int label = instances.get(0).label;
		for (int i = 1; i < instances.size(); i++){
			if (instances.get(i).label != label){
				isDone = false;
			}
		}
		return isDone;
	}
	
	public int getBestAttributeID(List<Instance> instances, ArrayList<Integer> attributeIDList){
		int noOfLabels = train.labels.size();
		double entropy = getEntropy(instances, noOfLabels);
		int bestAttributeID = -1;	
		double bestInfoGain = -1;
		
		//if (attributeIDList.size() == 0){
			//return null;
		//}
		
		for (int k = 0; k < attributeIDList.size(); k++){
			int attributeID = attributeIDList.get(k);
			String attribute = train.attributes.get(attributeID);
			int noOfAttributeLabel = attributeValues.get(attribute).size();
			double conditionalEntropy = getConditionalEntropy(instances, attributeID, noOfAttributeLabel, noOfLabels);
			double temp = entropy - conditionalEntropy;
			if (temp > bestInfoGain){
				bestInfoGain = temp;
				bestAttributeID = attributeID;
			}
		}
		return bestAttributeID;
			
	}
	


	/**
	 * Build a decision tree given a training set then prune it using a tuning
	 * set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */

	DecisionTreeImpl(DataSet train, DataSet tune) {
		// build a decision tree from train set
		this(train);
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: add code here
		double lastAccuracy = getAccuracy(tune.instances);
		DecTreeNode nodeCandiate = null;
		ArrayList<DecTreeNode> nonTerminalNodes = new ArrayList<DecTreeNode>();
		this.getNonLeafNode(root, nonTerminalNodes);
		double candidateAccuracy = -1;
		int candidateCount = 0;
		for (DecTreeNode currNode : nonTerminalNodes) {
			currNode.terminal = true;
			double accuracy = getAccuracy(tune.instances);
			currNode.terminal = false;
			int childrenCount = countDFS(currNode);
			if (accuracy > candidateAccuracy || (accuracy == candidateAccuracy && childrenCount > candidateCount)){
				candidateAccuracy = accuracy;
				candidateCount = childrenCount; 
				nodeCandiate = currNode;
			}
		}
		assert(nodeCandiate != null);
		if (candidateAccuracy >= lastAccuracy) {
			nodeCandiate.children = null;
			nodeCandiate.terminal = true;
		}
	}		
	public double getAccuracy(List<Instance> instances){
		if (instances.size() == 0){
			return 0;
		}
		int accuracyCount = 0;
		for (int i =0; i <instances.size(); i++){
			Instance instance = instances.get(i);
			int labelAssigned = train.labels.indexOf(this.classify(instance));
			if (labelAssigned == instance.label){
				accuracyCount ++;
			}
		}
		return accuracyCount*1.0/instances.size();
	}
	public void getNonLeafNode(DecTreeNode startNode, ArrayList<DecTreeNode> nonLeaf){
		DecTreeNode currentNode = startNode;
		if (currentNode.terminal){
			return;
		}
		if (!currentNode.terminal){
			nonLeaf.add(currentNode);
			int numberOfChildren = currentNode.children.size();
			for (int k = 0; k < numberOfChildren; k++){
				DecTreeNode childNode = currentNode.children.get(k);
				 getNonLeafNode(childNode, nonLeaf);
			}
		}
		return;
	}
	
	private int countDFS(DecTreeNode node){
		int count = 0;
		if (node.terminal == true) {
			assert(node.children == null || node.children.size() == 0);
			return 1;
		}
		for (DecTreeNode currNode : node.children) {
			count += countDFS(currNode);
		}
		return count;
	}

	
	
	//given a list of instances, this method returns the label that has the majority
	public int getMajority(List<Instance> instances, int noOfLabels){
		int majorityLabel = -1;
		int labelCount = 0;
		//if (instances == null){
			//return 0;
		//}
		int[] sorter = new int[noOfLabels];
		
		for (int i=0; i<instances.size(); i++){
			Instance instance = instances.get(i);
			sorter[instance.label]++;
		}
		for (int k = 0; k < noOfLabels; k++){
			if (labelCount < sorter[k]){
				labelCount = sorter[k];
				majorityLabel = k;
			}
		}
		return majorityLabel;
		
	}
	
	public double getEntropy(List<Instance> instances, int noOfLabels){
		double entropy = 0;
		if (instances == null){
			return 0;
		}
		int[] sorter = new int[noOfLabels];

		for (int i=0; i<instances.size(); i++){
			Instance instance = instances.get(i);
			sorter[instance.label]++;
		}
		for (int k = 0; k < noOfLabels; k++){
			if (sorter[k]==0) {
				return 0;
			}
			
			if (instances.size() > 0){
				double prob = sorter[k]*1.0/instances.size();
				entropy = entropy + prob * Math.log(prob)/Math.log(2)*1.0;
			}
		}
		entropy = -entropy;
		return entropy;
	}
	
	public double getConditionalEntropy(List<Instance> instances, int attributeID, int noOfAttributeLabel, int noOfLabels){
		double conditionalEntropy = 0;
		if (instances == null){
			return 0;
		}
		List<List<Instance>> subList = new ArrayList<List<Instance>>();
		for (int k = 0; k < noOfAttributeLabel; k++){
			List<Instance> temp = new ArrayList<Instance>();
			subList.add(temp);
		}
		
		int[] sortByAttr = new int[noOfAttributeLabel];
		for (int i=0; i<instances.size(); i++){
			Instance instance = instances.get(i);
			List<Integer> attributes = instance.attributes;
			int attributeLabel = attributes.get(attributeID);
			sortByAttr[attributeLabel]++;
			subList.get(attributeLabel).add(instance);
		}
		for (int k = 0; k < noOfAttributeLabel; k++){
			if (instances.size() > 0){
				double proportion = sortByAttr[k]*1.0/instances.size();
				List<Instance> sub = subList.get(k);
				conditionalEntropy = conditionalEntropy + proportion * this.getEntropy(sub, noOfLabels);
			}
		}
		return conditionalEntropy;
		
	}
	

	@Override
	public String classify(Instance instance) {

		// TODO: add code here
		DecTreeNode currentNode = root;
		int finalLabel = currentNode.label;
		
		
		while(!currentNode.terminal){
			DecTreeNode tempNode = currentNode;
			int currAttr = tempNode.attribute;
			int instanceAtributeLabel = instance.attributes.get(currAttr);
			int numberOfChildren = tempNode.children.size();
			for (int k = 0; k < numberOfChildren; k++){
				DecTreeNode childNode = tempNode.children.get(k);
				if (instanceAtributeLabel == childNode.parentAttributeValue){
					currentNode = childNode;
					finalLabel = childNode.label;
				}
			}
		}
		return train.labels.get(finalLabel);
		
	}

	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {

		printTreeNode(root, null, 0);
	}
	
	/**
	 * Prints the subtree of the node
	 * with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else{
			String parentAttribute = attributes.get(parent.attribute);
			value = attributeValues.get(parentAttribute).get(p.parentAttributeValue);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + labels.get(p.label) + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + attributes.get(p.attribute) + "?}");
			System.out.println(sb.toString());
			for(DecTreeNode child: p.children) {
				printTreeNode(child, p, k+1);
			}
		}
	}

	@Override
	public void rootInfoGain(DataSet train) {

		
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		int importantAttributeID = -1;
		// TODO: add code here
		double entropy = getEntropy(train.instances, train.labels.size());
		double infoGain = 0.0;
		for (int i = 0; i < train.attributes.size(); i++) { //For each attribute
			double conditionalEntropy = getConditionalEntropy(train.instances, i, 
					train.attributeValues.get(train.attributes.get(i)).size(), train.labels.size());
			double newGain = entropy - conditionalEntropy;
			//if (newGain > infoGain){
				//infoGain = newGain;
				//importantAttributeID = i;
			//}
			System.out.printf("%s %.5f\n", train.attributes.get(i) + " ", newGain);
		}
	}
	
	
	
	
}
