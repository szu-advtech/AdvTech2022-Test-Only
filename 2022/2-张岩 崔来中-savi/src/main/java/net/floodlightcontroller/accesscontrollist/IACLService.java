package net.floodlightcontroller.accesscontrollist;
import java.util.List;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IACLService extends IFloodlightService {
    public List<ACLRule> getRules();
    public boolean addRule(ACLRule rule);
    public void removeRule(int ruleid);
    public void removeAllRules();
}
