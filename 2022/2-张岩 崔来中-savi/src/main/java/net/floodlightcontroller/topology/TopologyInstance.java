package net.floodlightcontroller.topology;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import net.floodlightcontroller.routing.BroadcastTree;
import net.floodlightcontroller.routing.Link;
import net.floodlightcontroller.routing.Route;
import net.floodlightcontroller.routing.RouteId;
import net.floodlightcontroller.servicechaining.ServiceChain;
import net.floodlightcontroller.util.ClusterDFS;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.U64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
public class TopologyInstance {
    public static final short LT_SH_LINK = 1;
    public static final short LT_BD_LINK = 2;
    public static final short LT_TUNNEL  = 3;
    public static final int MAX_LINK_WEIGHT = 10000;
    public static final int MAX_PATH_WEIGHT = Integer.MAX_VALUE - MAX_LINK_WEIGHT - 1;
    public static final int PATH_CACHE_SIZE = 1000;
    protected static Logger log = LoggerFactory.getLogger(TopologyInstance.class);
    protected Set<NodePortTuple> blockedPorts;
    protected Set<Link> blockedLinks;
    protected Set<DatapathId> switches;
    protected Set<NodePortTuple> broadcastDomainPorts;
    protected Set<NodePortTuple> tunnelPorts;
    protected Map<DatapathId, BroadcastTree> destinationRootedTrees;
    protected Map<DatapathId, Set<NodePortTuple>> clusterPorts;
    protected Map<DatapathId, BroadcastTree> clusterBroadcastTrees;
    protected Map<DatapathId, Set<NodePortTuple>> clusterBroadcastNodePorts;
    protected BroadcastTree finiteBroadcastTree;
    protected Set<NodePortTuple> broadcastNodePorts;  
    protected Map<DatapathId, BroadcastTree> destinationRootedFullTrees;
    protected Map<NodePortTuple, Set<Link>> allLinks;
	protected Map<DatapathId, Set<OFPort>> allPorts;
    protected Map<DatapathId, Set<OFPort>> broadcastPortMap;
    protected class PathCacheLoader extends CacheLoader<RouteId, Route> {
        TopologyInstance ti;
        PathCacheLoader(TopologyInstance ti) {
            this.ti = ti;
        }
        @Override
        public Route load(RouteId rid) {
            return ti.buildroute(rid);
        }
    }
    private final PathCacheLoader pathCacheLoader = new PathCacheLoader(this);
    protected LoadingCache<RouteId, Route> pathcache;
    public TopologyInstance(Map<DatapathId, Set<OFPort>> switchPorts,
                            Set<NodePortTuple> blockedPorts,
                            Map<NodePortTuple, Set<Link>> switchPortLinks,
                            Set<NodePortTuple> broadcastDomainPorts,
                            Set<NodePortTuple> tunnelPorts, 
                            Map<NodePortTuple, Set<Link>> allLinks, 
                            Map<DatapathId, Set<OFPort>> allPorts) {
        this.switches = new HashSet<DatapathId>(switchPorts.keySet());
        this.switchPorts = new HashMap<DatapathId, Set<OFPort>>();
        for (DatapathId sw : switchPorts.keySet()) {
            this.switchPorts.put(sw, new HashSet<OFPort>(switchPorts.get(sw)));
        }
		this.allPorts = new HashMap<DatapathId, Set<OFPort>>();
		for (DatapathId sw : allPorts.keySet()) {
            this.allPorts.put(sw, new HashSet<OFPort>(allPorts.get(sw)));
        }
        this.blockedPorts = new HashSet<NodePortTuple>(blockedPorts);
        this.switchPortLinks = new HashMap<NodePortTuple, Set<Link>>();
        for (NodePortTuple npt : switchPortLinks.keySet()) {
            this.switchPortLinks.put(npt, new HashSet<Link>(switchPortLinks.get(npt)));
        }
		this.allLinks = new HashMap<NodePortTuple, Set<Link>>();
        for (NodePortTuple npt : allLinks.keySet()) {
            this.allLinks.put(npt, new HashSet<Link>(allLinks.get(npt)));
        }
        this.broadcastDomainPorts = new HashSet<NodePortTuple>(broadcastDomainPorts);
        this.tunnelPorts = new HashSet<NodePortTuple>(tunnelPorts);
        this.blockedLinks = new HashSet<Link>();
        this.clusters = new HashSet<Cluster>();
        this.switchClusterMap = new HashMap<DatapathId, Cluster>();
        this.destinationRootedTrees = new HashMap<DatapathId, BroadcastTree>();
        this.destinationRootedFullTrees= new HashMap<DatapathId, BroadcastTree>();
		this.broadcastNodePorts= new HashSet<NodePortTuple>();
		this.broadcastPortMap = new HashMap<DatapathId,Set<OFPort>>();
        this.clusterBroadcastTrees = new HashMap<DatapathId, BroadcastTree>();
        this.clusterBroadcastNodePorts = new HashMap<DatapathId, Set<NodePortTuple>>();
        pathcache = CacheBuilder.newBuilder().concurrencyLevel(4)
                    .maximumSize(1000L)
                    .build(
                            new CacheLoader<RouteId, Route>() {
                                public Route load(RouteId rid) {
                                    return pathCacheLoader.load(rid);
                                }
                            });
    }
    public void compute() {
        identifyOpenflowDomains();
        addLinksToOpenflowDomains();
        calculateShortestPathTreeInClusters();
        calculateBroadcastNodePortsInClusters();
		calculateAllShortestPaths();
        calculateAllBroadcastNodePorts();
       	calculateBroadcastPortMap();
        printTopology();
    }
    public boolean isEdge(DatapathId sw, OFPort portId) { 
		NodePortTuple np = new NodePortTuple(sw, portId);
		if (allLinks.get(np) == null || allLinks.get(np).isEmpty()) {
			return true;
		}
		else {
			return false;
		}
    }   
    public Set<OFPort> swBroadcastPorts(DatapathId sw){
    	return this.broadcastPortMap.get(sw);
    }
    public void printTopology() {
        log.debug("-----------------Topology-----------------------");
        log.debug("All Links: {}", allLinks);
		log.debug("Broadcast Tree: {}", finiteBroadcastTree);
        log.debug("Broadcast Domain Ports: {}", broadcastDomainPorts);
        log.debug("Tunnel Ports: {}", tunnelPorts);
        log.debug("Clusters: {}", clusters);
        log.debug("Destination Rooted Full Trees: {}", destinationRootedFullTrees);
        log.debug("Broadcast Node Ports: {}", broadcastNodePorts);
        log.debug("-----------------------------------------------");  
    }
    protected void addLinksToOpenflowDomains() {
        for(DatapathId s: switches) {
            if (switchPorts.get(s) == null) continue;
            for (OFPort p: switchPorts.get(s)) {
                NodePortTuple np = new NodePortTuple(s, p);
                if (switchPortLinks.get(np) == null) continue;
                if (isBroadcastDomainPort(np)) continue;
                for(Link l: switchPortLinks.get(np)) {
                    if (isBlockedLink(l)) continue;
                    if (isBroadcastDomainLink(l)) continue;
                    Cluster c1 = switchClusterMap.get(l.getSrc());
                    Cluster c2 = switchClusterMap.get(l.getDst());
                    if (c1 ==c2) {
                        c1.addLink(l);
                    }
                }
            }
        }
    }
    public void identifyOpenflowDomains() {
        Map<DatapathId, ClusterDFS> dfsList = new HashMap<DatapathId, ClusterDFS>();
        if (switches == null) return;
        for (DatapathId key : switches) {
            ClusterDFS cdfs = new ClusterDFS();
            dfsList.put(key, cdfs);
        }
        Set<DatapathId> currSet = new HashSet<DatapathId>();
        for (DatapathId sw : switches) {
            ClusterDFS cdfs = dfsList.get(sw);
            if (cdfs == null) {
                log.error("No DFS object for switch {} found.", sw);
            } else if (!cdfs.isVisited()) {
                dfsTraverse(0, 1, sw, dfsList, currSet);
            }
        }
    }
    private long dfsTraverse (long parentIndex, long currIndex, DatapathId currSw,
                              Map<DatapathId, ClusterDFS> dfsList, Set<DatapathId> currSet) {
        ClusterDFS currDFS = dfsList.get(currSw);
        Set<DatapathId> nodesInMyCluster = new HashSet<DatapathId>();
        Set<DatapathId> myCurrSet = new HashSet<DatapathId>();
        currDFS.setVisited(true);
        currDFS.setDfsIndex(currIndex);
        currDFS.setParentDFSIndex(parentIndex);
        currIndex++;
        if (switchPorts.get(currSw) != null){
            for (OFPort p : switchPorts.get(currSw)) {
                Set<Link> lset = switchPortLinks.get(new NodePortTuple(currSw, p));
                if (lset == null) continue;
                for (Link l : lset) {
                    DatapathId dstSw = l.getDst();
                    if (dstSw.equals(currSw)) continue;
                    if (switchClusterMap.get(dstSw) != null) continue;
                    if (isBlockedLink(l)) continue;
                    if (isBroadcastDomainLink(l)) continue;
                    ClusterDFS dstDFS = dfsList.get(dstSw);
                    if (dstDFS.getDfsIndex() < currDFS.getDfsIndex()) {
                        if (dstDFS.getDfsIndex() < currDFS.getLowpoint()) {
                            currDFS.setLowpoint(dstDFS.getDfsIndex());
                        }
                    } else if (!dstDFS.isVisited()) {
                        currIndex = dfsTraverse(
                        		currDFS.getDfsIndex(), 
                        		currIndex, dstSw, 
                        		dfsList, myCurrSet);
                        if (currIndex < 0) return -1;
                        if (dstDFS.getLowpoint() < currDFS.getLowpoint()) {
                            currDFS.setLowpoint(dstDFS.getLowpoint());
                        }
                        nodesInMyCluster.addAll(myCurrSet);
                        myCurrSet.clear();
                    }
                }
            }
        }
        nodesInMyCluster.add(currSw);
        currSet.addAll(nodesInMyCluster);
        if (currDFS.getLowpoint() > currDFS.getParentDFSIndex()) {
            Cluster sc = new Cluster();
            for (DatapathId sw : currSet) {
                sc.add(sw);
                switchClusterMap.put(sw, sc);
            }
            currSet.clear();
            clusters.add(sc);
        }
        return currIndex;
    }
    public Set<NodePortTuple> getBlockedPorts() {
        return this.blockedPorts;
    }
    protected Set<Link> getBlockedLinks() {
        return this.blockedLinks;
    }
    protected boolean isBlockedLink(Link l) {
        NodePortTuple n1 = new NodePortTuple(l.getSrc(), l.getSrcPort());
        NodePortTuple n2 = new NodePortTuple(l.getDst(), l.getDstPort());
        return (isBlockedPort(n1) || isBlockedPort(n2));
    }
    protected boolean isBlockedPort(NodePortTuple npt) {
        return blockedPorts.contains(npt);
    }
    protected boolean isTunnelPort(NodePortTuple npt) {
        return tunnelPorts.contains(npt);
    }
    protected boolean isTunnelLink(Link l) {
        NodePortTuple n1 = new NodePortTuple(l.getSrc(), l.getSrcPort());
        NodePortTuple n2 = new NodePortTuple(l.getDst(), l.getDstPort());
        return (isTunnelPort(n1) || isTunnelPort(n2));
    }
    public boolean isBroadcastDomainLink(Link l) {
        NodePortTuple n1 = new NodePortTuple(l.getSrc(), l.getSrcPort());
        NodePortTuple n2 = new NodePortTuple(l.getDst(), l.getDstPort());
        return (isBroadcastDomainPort(n1) || isBroadcastDomainPort(n2));
    }
    public boolean isBroadcastDomainPort(NodePortTuple npt) {
        return broadcastDomainPorts.contains(npt);
    }
    protected class NodeDist implements Comparable<NodeDist> {
        private final DatapathId node;
        public DatapathId getNode() {
            return node;
        }
        private final int dist;
        public int getDist() {
            return dist;
        }
        public NodeDist(DatapathId node, int dist) {
            this.node = node;
            this.dist = dist;
        }
        @Override
        public int compareTo(NodeDist o) {
            if (o.dist == this.dist) {
                return (int)(this.node.getLong() - o.node.getLong());
            }
            return this.dist - o.dist;
        }
        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null)
                return false;
            if (getClass() != obj.getClass())
                return false;
            NodeDist other = (NodeDist) obj;
            if (!getOuterType().equals(other.getOuterType()))
                return false;
            if (node == null) {
                if (other.node != null)
                    return false;
            } else if (!node.equals(other.node))
                return false;
            return true;
        }
        @Override
        public int hashCode() {
            assert false : "hashCode not designed";
            return 42;
        }
        private TopologyInstance getOuterType() {
            return TopologyInstance.this;
        }
    }
    protected BroadcastTree clusterDijkstra(Cluster c, DatapathId root,
                                     Map<Link, Integer> linkCost,
                                     boolean isDstRooted) {
        HashMap<DatapathId, Link> nexthoplinks = new HashMap<DatapathId, Link>();
        HashMap<DatapathId, Integer> cost = new HashMap<DatapathId, Integer>();
        int w;
        for (DatapathId node : c.links.keySet()) {
            nexthoplinks.put(node, null);
            cost.put(node, MAX_PATH_WEIGHT);
        }
        HashMap<DatapathId, Boolean> seen = new HashMap<DatapathId, Boolean>();
        PriorityQueue<NodeDist> nodeq = new PriorityQueue<NodeDist>();
        nodeq.add(new NodeDist(root, 0));
        cost.put(root, 0);
        while (nodeq.peek() != null) {
            NodeDist n = nodeq.poll();
            DatapathId cnode = n.getNode();
            int cdist = n.getDist();
            if (cdist >= MAX_PATH_WEIGHT) break;
            if (seen.containsKey(cnode)) continue;
            seen.put(cnode, true);
            for (Link link : c.links.get(cnode)) {
                DatapathId neighbor;
                if (isDstRooted == true) {
                	neighbor = link.getSrc();
                } else {
                	neighbor = link.getDst();
                }
                if (neighbor.equals(cnode)) continue;
                if (seen.containsKey(neighbor)) continue;
                if (linkCost == null || linkCost.get(link) == null) {
                	w = 1;
                } else {
                	w = linkCost.get(link);
                }
                if (ndist < cost.get(neighbor)) {
                    cost.put(neighbor, ndist);
                    nexthoplinks.put(neighbor, link);
                    NodeDist ndTemp = new NodeDist(neighbor, ndist);
                    nodeq.remove(ndTemp);
                    nodeq.add(ndTemp);
                }
            }
        }
        BroadcastTree ret = new BroadcastTree(nexthoplinks, cost);
        return ret;
    }
    protected BroadcastTree dijkstra(Map<DatapathId, Set<Link>> links, DatapathId root,
            Map<Link, Integer> linkCost,
            boolean isDstRooted) {
    	HashMap<DatapathId, Link> nexthoplinks = new HashMap<DatapathId, Link>();
    	HashMap<DatapathId, Integer> cost = new HashMap<DatapathId, Integer>();
    	int w;
    	for (DatapathId node : links.keySet()) {
    		nexthoplinks.put(node, null);
    		cost.put(node, MAX_PATH_WEIGHT);
    	}
    	HashMap<DatapathId, Boolean> seen = new HashMap<DatapathId, Boolean>();
    	PriorityQueue<NodeDist> nodeq = new PriorityQueue<NodeDist>();
    	nodeq.add(new NodeDist(root, 0));
    	cost.put(root, 0);
    	while (nodeq.peek() != null) {
    		NodeDist n = nodeq.poll();
    		DatapathId cnode = n.getNode();
    		int cdist = n.getDist();
    		if (cdist >= MAX_PATH_WEIGHT) break;
    		if (seen.containsKey(cnode)) continue;
    		seen.put(cnode, true);
    		for (Link link : links.get(cnode)) {
    			DatapathId neighbor;
    			if (isDstRooted == true) {
    				neighbor = link.getSrc();
    			} else {
    				neighbor = link.getDst();
    			}
    			if (neighbor.equals(cnode)) continue;
    			if (seen.containsKey(neighbor)) continue;
    			if (linkCost == null || linkCost.get(link) == null) {
    				w = 1;
    			} else {
    				w = linkCost.get(link);
    			}
    			if (ndist < cost.get(neighbor)) {
    				cost.put(neighbor, ndist);
    				nexthoplinks.put(neighbor, link);
    				NodeDist ndTemp = new NodeDist(neighbor, ndist);
    				nodeq.remove(ndTemp);
    				nodeq.add(ndTemp);
    			}
    		}
    	}
    	BroadcastTree ret = new BroadcastTree(nexthoplinks, cost);
		return ret;
	}
    public void calculateAllShortestPaths() {
    	this.broadcastNodePorts.clear();
    	this.destinationRootedFullTrees.clear();
    	Map<Link, Integer> linkCost = new HashMap<Link, Integer>();
        int tunnel_weight = switchPorts.size() + 1;
        for (NodePortTuple npt : tunnelPorts) {
            if (allLinks.get(npt) == null) continue;
            for (Link link : allLinks.get(npt)) {
                if (link == null) continue;
                linkCost.put(link, tunnel_weight);
            }
        }
        Map<DatapathId, Set<Link>> linkDpidMap = new HashMap<DatapathId, Set<Link>>();
        for (DatapathId s : switches) {
            if (switchPorts.get(s) == null) continue;
            for (OFPort p : switchPorts.get(s)) {
                NodePortTuple np = new NodePortTuple(s, p);
                if (allLinks.get(np) == null) continue;
                for (Link l : allLinks.get(np)) {
                	if (linkDpidMap.containsKey(s)) {
                		linkDpidMap.get(s).add(l);
                	}
                	else {
                		linkDpidMap.put(s, new HashSet<Link>(Arrays.asList(l)));
                	}
                }
            }
        }   
        for (DatapathId node : linkDpidMap.keySet()) {
        	BroadcastTree tree = dijkstra(linkDpidMap, node, linkCost, true);
            destinationRootedFullTrees.put(node, tree);
        }
        if (this.destinationRootedFullTrees.size() > 0) {
			this.finiteBroadcastTree = destinationRootedFullTrees.values().iterator().next();
        }         	
    }
    protected void calculateShortestPathTreeInClusters() {
        pathcache.invalidateAll();
        destinationRootedTrees.clear();
        Map<Link, Integer> linkCost = new HashMap<Link, Integer>();
        int tunnel_weight = switchPorts.size() + 1;
        for (NodePortTuple npt : tunnelPorts) {
            if (switchPortLinks.get(npt) == null) continue;
            for (Link link : switchPortLinks.get(npt)) {
                if (link == null) continue;
                linkCost.put(link, tunnel_weight);
            }
        }
        for (Cluster c : clusters) {
            for (DatapathId node : c.links.keySet()) {
                BroadcastTree tree = clusterDijkstra(c, node, linkCost, true);
                destinationRootedTrees.put(node, tree);
            }
        }
    }
    protected void calculateBroadcastTreeInClusters() {
        for(Cluster c: clusters) {
            BroadcastTree tree = destinationRootedTrees.get(c.id);
            clusterBroadcastTrees.put(c.id, tree);
        }
    }
	protected Set<NodePortTuple> getAllBroadcastNodePorts() {
		return this.broadcastNodePorts;
	}
    protected void calculateAllBroadcastNodePorts() {
		if (this.destinationRootedFullTrees.size() > 0) {
			this.finiteBroadcastTree = destinationRootedFullTrees.values().iterator().next();
			Map<DatapathId, Link> links = finiteBroadcastTree.getLinks();
			if (links == null) return;
			for (DatapathId nodeId : links.keySet()) {
				Link l = links.get(nodeId);
				if (l == null) continue;
				NodePortTuple npt1 = new NodePortTuple(l.getSrc(), l.getSrcPort());
				NodePortTuple npt2 = new NodePortTuple(l.getDst(), l.getDstPort());
				this.broadcastNodePorts.add(npt1);
				this.broadcastNodePorts.add(npt2);
			}    
		}		
    }
    protected void calculateBroadcastPortMap(){
		this.broadcastPortMap.clear();
		for (DatapathId sw : this.switches) {
			for (OFPort p : this.allPorts.get(sw)){
				NodePortTuple npt = new NodePortTuple(sw, p);
				if (isEdge(sw, p) || broadcastNodePorts.contains(npt)) { 
					if (broadcastPortMap.containsKey(sw)) {
                		broadcastPortMap.get(sw).add(p);
                	} else {
                		broadcastPortMap.put(sw, new HashSet<OFPort>(Arrays.asList(p)));
                	}
				}      		
			}
		}
    }
    protected void calculateBroadcastNodePortsInClusters() {
        clusterBroadcastTrees.clear();
        calculateBroadcastTreeInClusters();
        for (Cluster c : clusters) {
            BroadcastTree tree = clusterBroadcastTrees.get(c.id);
            Set<NodePortTuple> nptSet = new HashSet<NodePortTuple>();
            Map<DatapathId, Link> links = tree.getLinks();
            if (links == null) continue;
            for (DatapathId nodeId : links.keySet()) {
                Link l = links.get(nodeId);
                if (l == null) continue;
                NodePortTuple npt1 = new NodePortTuple(l.getSrc(), l.getSrcPort());
                NodePortTuple npt2 = new NodePortTuple(l.getDst(), l.getDstPort());
                nptSet.add(npt1);
                nptSet.add(npt2);
            }
            clusterBroadcastNodePorts.put(c.id, nptSet);
        }
    }
    protected Route buildroute(RouteId id) {
        NodePortTuple npt;
        DatapathId srcId = id.getSrc();
        DatapathId dstId = id.getDst();
        LinkedList<NodePortTuple> sPorts = new LinkedList<NodePortTuple>();
        if (destinationRootedFullTrees == null) return null;
        if (destinationRootedFullTrees.get(dstId) == null) return null;
        Map<DatapathId, Link> nexthoplinks = destinationRootedFullTrees.get(dstId).getLinks();
        if (!switches.contains(srcId) || !switches.contains(dstId)) {
            log.info("buildroute: Standalone switch: {}", srcId);
        } else if ((nexthoplinks!=null) && (nexthoplinks.get(srcId) != null)) {
            while (!srcId.equals(dstId)) {
                Link l = nexthoplinks.get(srcId);
                npt = new NodePortTuple(l.getSrc(), l.getSrcPort());
                sPorts.addLast(npt);
                npt = new NodePortTuple(l.getDst(), l.getDstPort());
                sPorts.addLast(npt);
                srcId = nexthoplinks.get(srcId).getDst();
            }
        }
        Route result = null;
        if (sPorts != null && !sPorts.isEmpty()) {
            result = new Route(id, sPorts);
        }
        if (log.isTraceEnabled()) {
            log.trace("buildroute: {}", result);
        }
        return result;
    }
    protected int getCost(DatapathId srcId, DatapathId dstId) {
        BroadcastTree bt = destinationRootedTrees.get(dstId);
        if (bt == null) return -1;
        return bt.getCost(srcId);
    }
    protected Set<Cluster> getClusters() {
        return clusters;
    }
    protected boolean routeExists(DatapathId srcId, DatapathId dstId) {
        BroadcastTree bt = destinationRootedTrees.get(dstId);
        if (bt == null) return false;
        Link link = bt.getLinks().get(srcId);
        if (link == null) return false;
        return true;
    }
    protected Route getRoute(ServiceChain sc, DatapathId srcId, OFPort srcPort,
            DatapathId dstId, OFPort dstPort, U64 cookie) {
        if (srcId.equals(dstId) && srcPort.equals(dstPort)) {
            return null;
        }
        List<NodePortTuple> nptList;
        NodePortTuple npt;
        Route r = getRoute(srcId, dstId, U64.of(0));
        if (r == null && !srcId.equals(dstId)) {
        	return null;
        }
        if (r != null) {
            nptList= new ArrayList<NodePortTuple>(r.getPath());
        } else {
            nptList = new ArrayList<NodePortTuple>();
        }
        npt = new NodePortTuple(srcId, srcPort);
        npt = new NodePortTuple(dstId, dstPort);
        RouteId id = new RouteId(srcId, dstId);
        r = new Route(id, nptList);
        return r;
    }
    protected Route getRoute(DatapathId srcId, DatapathId dstId, U64 cookie) {
        if (srcId.equals(dstId)) return null;
        RouteId id = new RouteId(srcId, dstId);
        Route result = null;
        try {
            result = pathcache.get(id);
        } catch (Exception e) {
            log.error("{}", e);
        }
        if (log.isTraceEnabled()) {
            log.trace("getRoute: {} -> {}", id, result);
        }
        return result;
    }
    protected BroadcastTree getBroadcastTreeForCluster(long clusterId){
        Cluster c = switchClusterMap.get(clusterId);
        if (c == null) return null;
        return clusterBroadcastTrees.get(c.id);
    }
    protected boolean isInternalToOpenflowDomain(DatapathId switchid, OFPort port) {
        return !isAttachmentPointPort(switchid, port);
    }
    public boolean isAttachmentPointPort(DatapathId switchid, OFPort port) {
        NodePortTuple npt = new NodePortTuple(switchid, port);
        if (switchPortLinks.containsKey(npt)) return false;
        return true;
    }
    protected DatapathId getOpenflowDomainId(DatapathId switchId) {
        Cluster c = switchClusterMap.get(switchId);
        if (c == null) return switchId;
        return c.getId();
    }
    protected DatapathId getL2DomainId(DatapathId switchId) {
        return getOpenflowDomainId(switchId);
    }
    protected Set<DatapathId> getSwitchesInOpenflowDomain(DatapathId switchId) {
        Cluster c = switchClusterMap.get(switchId);
        if (c == null) {
            Set<DatapathId> nodes = new HashSet<DatapathId>();
            nodes.add(switchId);
            return nodes;
        }
        return (c.getNodes());
    }
    protected boolean inSameOpenflowDomain(DatapathId switch1, DatapathId switch2) {
        Cluster c1 = switchClusterMap.get(switch1);
        Cluster c2 = switchClusterMap.get(switch2);
        if (c1 != null && c2 != null)
            return (c1.getId().equals(c2.getId()));
        return (switch1.equals(switch2));
    }
    public boolean isAllowed(DatapathId sw, OFPort portId) {
        return true;
    }
    protected boolean isIncomingBroadcastAllowedOnSwitchPort(DatapathId sw, OFPort portId) {
        if (!isEdge(sw, portId)){       
            NodePortTuple npt = new NodePortTuple(sw, portId);
            if (broadcastNodePorts.contains(npt))
                return true;
            else return false;
        }
        return true;
    }
    public boolean isConsistent(DatapathId oldSw, OFPort oldPort, DatapathId newSw, OFPort newPort) {
        if (isInternalToOpenflowDomain(newSw, newPort)) return true;
        return (oldSw.equals(newSw) && oldPort.equals(newPort));
    }
    protected Set<NodePortTuple> getBroadcastNodePortsInCluster(DatapathId sw) {
        DatapathId clusterId = getOpenflowDomainId(sw);
        return clusterBroadcastNodePorts.get(clusterId);
    }
    public boolean inSameBroadcastDomain(DatapathId s1, OFPort p1, DatapathId s2, OFPort p2) {
        return false;
    }
    public boolean inSameL2Domain(DatapathId switch1, DatapathId switch2) {
        return inSameOpenflowDomain(switch1, switch2);
    }
    public NodePortTuple getOutgoingSwitchPort(DatapathId src, OFPort srcPort,
            DatapathId dst, OFPort dstPort) {
        return new NodePortTuple(dst, dstPort);
    }
    public NodePortTuple getIncomingSwitchPort(DatapathId src, OFPort srcPort,
            DatapathId dst, OFPort dstPort) {
        return new NodePortTuple(src, srcPort);
    }
    public Set<DatapathId> getSwitches() {
        return switches;
    }
    public Set<OFPort> getPortsWithLinks(DatapathId sw) {
        return switchPorts.get(sw);
    }
    public Set<OFPort> getBroadcastPorts(DatapathId targetSw, DatapathId src, OFPort srcPort) {
        Set<OFPort> result = new HashSet<OFPort>();
        DatapathId clusterId = getOpenflowDomainId(targetSw);
        for (NodePortTuple npt : clusterPorts.get(clusterId)) {
            if (npt.getNodeId().equals(targetSw)) {
                result.add(npt.getPortId());
            }
        }
        return result;
    }
    public NodePortTuple getAllowedOutgoingBroadcastPort(DatapathId src, OFPort srcPort, DatapathId dst, OFPort dstPort) {
        return null;
    }
    public NodePortTuple getAllowedIncomingBroadcastPort(DatapathId src, OFPort srcPort) {
        return null;
    }
}