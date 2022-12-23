package net.floodlightcontroller.accesscontrollist;
import static org.junit.Assert.assertEquals;
import java.io.IOException;
import net.floodlightcontroller.accesscontrollist.ACL;
import net.floodlightcontroller.accesscontrollist.ACLRule;
import net.floodlightcontroller.accesscontrollist.IACLService;
import net.floodlightcontroller.accesscontrollist.util.IPAddressUtil;
import net.floodlightcontroller.accesscontrollist.web.ACLRuleResource;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.test.FloodlightTestCase;
import org.junit.Test;
import org.restlet.Context;
public class ACLRuleResourceTest extends FloodlightTestCase {
	@Test
	public void testJsonToRule(){
		String json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1/32\",\"dst-ip\": \"10.0.0.2/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
		try {
			ACLRule rule = ACLRuleResource.jsonToRule(json);
			assertEquals(rule.getNw_src(),"10.0.0.1/32");
			assertEquals(rule.getNw_dst(),"10.0.0.2/32");
			int[] cidr = IPAddressUtil.parseCIDR("10.0.0.1/32");
			assertEquals(rule.getNw_src_prefix(), cidr[0]);
			assertEquals(rule.getNw_src_maskbits(), cidr[1]);
			cidr = IPAddressUtil.parseCIDR("10.0.0.2/32");
			assertEquals(rule.getNw_dst_prefix(), cidr[0]);
			assertEquals(rule.getNw_dst_maskbits(), cidr[1]);
			assertEquals(rule.getNw_proto(),6);
			assertEquals(rule.getTp_dst(), 80);
		} catch (IOException e) {
			e.printStackTrace();
		}
		json = "{\"nw-prot\":\"TCP\",\"src-ip\":\"10.0.0.1/32\",\"dst-ip\": \"10.0.0.2/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
		try {
			ACLRule rule = ACLRuleResource.jsonToRule(json);
			assertEquals(rule.getNw_src(),"10.0.0.1/32");
			assertEquals(rule.getNw_dst(),"10.0.0.2/32");
			int[] cidr = IPAddressUtil.parseCIDR("10.0.0.1/32");
			assertEquals(rule.getNw_src_prefix(), cidr[0]);
			assertEquals(rule.getNw_src_maskbits(), cidr[1]);
			cidr = IPAddressUtil.parseCIDR("10.0.0.2/32");
			assertEquals(rule.getNw_dst_prefix(), cidr[0]);
			assertEquals(rule.getNw_dst_maskbits(), cidr[1]);
			assertEquals(rule.getNw_proto(),0);
			assertEquals(rule.getTp_dst(), 0);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	@Test
	public void testStore(){
		ACL s = new ACL();
        FloodlightModuleContext fmc = new FloodlightModuleContext();
		try {
			s.init(fmc);
		} catch (FloodlightModuleException e) {
			e.printStackTrace();
		}
		ACLRuleResource r = new ACLRuleResource();
		Context ctx = new Context();
		r.init(ctx, null, null);
		r.getContext().getAttributes().putIfAbsent(IACLService.class.getCanonicalName(), s);
        String json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1/32\",\"dst-ip\": \"10.0.0.2/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Success! New rule added." + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1/32\",\"dst-ip\": \"10.0.0.2/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! The new ACL rule matches an existing rule." + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1/8\",\"dst-ip\": \"10.0.0.2/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Success! New rule added." + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.2/32\",\"dst-ip\": \"10.0.0.2/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! The new ACL rule matches an existing rule." + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! Either nw_src or nw_dst must be specified." + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! " + "CIDR mask bits must be specified." + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.256/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! Octet values in specified IPv4 address must be 0 <= value <= 255" + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.01/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! Specified IPv4 address mustcontain 4 sets of numerical digits separated by periods" + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1/a\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! CIDR mask bits must be specified as a number(0 ~ 32)." + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1/33\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! CIDR mask bits must be 0 <= value <= 32." + "\"}");
        json = "{\"nw-proto\":\"ARP\",\"src-ip\":\"10.0.0.1/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! nw-proto must be specified as (TCP || UDP || ICMP)." + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1/32\",\"tp-dst\":\"a\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! tp-dst must be specified as a number." + "\"}");
        json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1/32\",\"tp-dst\":\"80\",\"action\":\"PERMIT\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Failed! action must be specidied as (allow || deny)." + "\"}");
	}
	@Test
	public void testRemove(){
		ACL s = new ACL();
        FloodlightModuleContext fmc = new FloodlightModuleContext();
		try {
			s.init(fmc);
		} catch (FloodlightModuleException e) {
			e.printStackTrace();
		}
		ACLRuleResource r = new ACLRuleResource();
		Context ctx = new Context();
		r.init(ctx, null, null);
		r.getContext().getAttributes().putIfAbsent(IACLService.class.getCanonicalName(), s);
        String json = "{\"nw-proto\":\"TCP\",\"src-ip\":\"10.0.0.1/32\",\"dst-ip\": \"10.0.0.2/32\",\"tp-dst\":\"80\",\"action\":\"ALLOW\"}";
        assertEquals(r.store(json),"{\"status\" : \"" + "Success! New rule added." + "\"}");
        json = "{\"ruleid\":\"a\"}";
        assertEquals(r.remove(json),"{\"status\" : \"" + "Failed! ruleid must be specified as a number." + "\"}");
        json = "{\"ruleid\":\"2\"}";
        assertEquals(r.remove(json),"{\"status\" : \"" + "Failed! a rule with this ID doesn't exist." + "\"}");
        json = "{\"ruleid\":\"1\"}";
        assertEquals(r.remove(json),"{\"status\" : \"" + "Success! Rule deleted" + "\"}");
	}
}
